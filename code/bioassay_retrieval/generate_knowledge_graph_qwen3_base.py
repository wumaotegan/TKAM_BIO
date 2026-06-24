

from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor

from llm import LLMGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Optional, Any
from dashscope import Generation
from bs4 import BeautifulSoup
from datetime import datetime
from http import HTTPStatus
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import hashlib
import logging
from functools import partial
#import fitz
import time
import os
import re
import dashscope
import warnings
from multiprocessing import Pool
import dashscope
from http import HTTPStatus
from langchain_openai import OpenAIEmbeddings
warnings.filterwarnings("ignore")
# from magic_pdf.pipe.TXTPipe import TXTPipe
# from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
from pathlib import Path
class sourceNode:
    file_name:str=None
    file_size:int=None
    file_type:str=None
    file_source:str=None
    status:str=None
    url:str=None
    gcsBucket:str=None
    gcsBucketFolder:str=None
    gcsProjectId:str=None
    awsAccessKeyId:str=None
    node_count:int=None
    relationship_count:str=None
    model:str=None
    created_at:datetime=None
    updated_at:datetime=None
    processing_time:float=None
    error_message:str=None
    total_pages:int=None
    total_chunks:int=None
    language:str=None
    is_cancelled:bool=None
    processed_chunk:int=None
    access_token:str=None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MERGED_DIR =  PROJECT_ROOT /"data/bioassays_description/aid_test_100"

uri ='bolt://localhost:7687'
userName ='neo4j'
password ='12345678'

API_KEY = "your_api_key" 
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
dashscope.api_key = API_KEY

MODEL_VERSIONS = {"qwen": "qwen1.5-72b-chat",
                  "qwen3-max": "qwen3-max"}

model ='qwen3-max'


if model not in MODEL_VERSIONS:
    raise ValueError(f"Unsupported model: {model}. Available models: {list(MODEL_VERSIONS.keys())}")

UPDATE_GRAPH_CHUNKS_PROCESSED = 20
NUMBER_OF_CHUNKS_TO_COMBINE = 6
KNN_MIN_SCORE = "0.8"
IS_EMBEDDING="True"

USE_FUNCTION_CALL=True
EMBEDDING_MODEL = "text-embedding-v4"


def load_embedding_model(embedding_model_name):
    if embedding_model_name == "BCE":
        embedding_model_name = '../bce-embedding-base_v1'
        embedding_model_kwargs = {'device': 'cuda:0'}
        embedding_encode_kwargs = {'batch_size': 32, 'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=embedding_model_kwargs,
            encode_kwargs=embedding_encode_kwargs
        )
        dimension = 768
        return embeddings, dimension
    elif embedding_model_name == "text-embedding-v4":
        embeddings = OpenAIEmbeddings(
            api_key=API_KEY,
            base_url=API_URL,
            model="text-embedding-v4",
            check_embedding_ctx_length=False,
            dimensions=2048
        )
        dimension = 2048
        return embeddings, dimension

CHAT_SEARCH_KWARG_SCORE_THRESHOLD = 0.7
CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD =0.1
USING_HTML_PRE= False
CHAT_SEARCH_KWARG_K = 5
CHAT_DOC_SPLIT_SIZE = 2000

VECTOR_SEARCH_QUERY = """
WITH node AS chunk, score
MATCH (chunk)-[:PART_OF]->(d:Document)
WITH d, collect(distinct {chunk: chunk, score: score}) as chunks, avg(score) as avg_score
WITH d, avg_score, 
     [c in chunks | c.chunk.text] as texts, 
     [c in chunks | {id: c.chunk.id, score: c.score}] as chunkdetails
WITH d, avg_score, chunkdetails,
     apoc.text.join(texts, "\n----\n") as text
RETURN text, avg_score AS score, 
       {source: d.fileName, chunkdetails: chunkdetails} as metadata
"""

VECTOR_GRAPH_SEARCH_QUERY = """
WITH node as chunk, score
// find the document of the chunk
MATCH (chunk)-[:PART_OF]->(d:Document)
// fetch entities
CALL { WITH chunk
// entities connected to the chunk
// todo only return entities that are actually in the chunk, remember we connect all extracted entities to all chunks
MATCH (chunk)-[:HAS_ENTITY]->(e)

// depending on match to query embedding either 1 or 2 step expansion
WITH CASE WHEN true // vector.similarity.cosine($embedding, e.embedding ) <= 0.95
THEN 
collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,1}(:!Chunk&!Document) RETURN path }
ELSE 
collect { MATCH path=(e)(()-[rels:!HAS_ENTITY&!PART_OF]-()){0,2}(:!Chunk&!Document) RETURN path } 
END as paths

RETURN collect{ unwind paths as p unwind relationships(p) as r return distinct r} as rels,
collect{ unwind paths as p unwind nodes(p) as n return distinct n} as nodes
}
// aggregate chunk-details and de-duplicate nodes and relationships
WITH d, collect(DISTINCT {chunk: chunk, score: score}) AS chunks, avg(score) as avg_score, apoc.coll.toSet(apoc.coll.flatten(collect(rels))) as rels,

// TODO sort by relevancy (embeddding comparision?) cut off after X (e.g. 25) nodes?
apoc.coll.toSet(apoc.coll.flatten(collect(
                [r in rels |[startNode(r),endNode(r)]]),true)) as nodes

// generate metadata and text components for chunks, nodes and relationships
WITH d, avg_score,
     [c IN chunks | c.chunk.text] AS texts, 
     [c IN chunks | {id: c.chunk.id, score: c.score}] AS chunkdetails,  
  apoc.coll.sort([n in nodes | 

coalesce(apoc.coll.removeAll(labels(n),['__Entity__'])[0],"") +":"+ 
n.id + (case when n.description is not null then " ("+ n.description+")" else "" end)]) as nodeTexts,
	apoc.coll.sort([r in rels 
    // optional filter if we limit the node-set
    // WHERE startNode(r) in nodes AND endNode(r) in nodes 
  | 
coalesce(apoc.coll.removeAll(labels(startNode(r)),['__Entity__'])[0],"") +":"+ 
startNode(r).id +
" " + type(r) + " " + 
coalesce(apoc.coll.removeAll(labels(endNode(r)),['__Entity__'])[0],"") +":" + 
endNode(r).id
]) as relTexts

// combine texts into response-text
WITH d, avg_score,chunkdetails,
"Text Content:\n" +
apoc.text.join(texts,"\n----\n") +
"\n----\nEntities:\n"+
apoc.text.join(nodeTexts,"\n") +
"\n----\nRelationships:\n"+
apoc.text.join(relTexts,"\n")

as text
RETURN text, avg_score as score, {length:size(text), source: d.fileName, chunkdetails: chunkdetails} AS metadata
"""


def get_llm(model_version: str):

    llm = ChatOpenAI(api_key=API_KEY,
                        base_url=API_URL,
                        model=model_version,
                        top_p=0.8,
                        temperature=0.6)

    logging.info(f"Model created - Model Version: {model_version}")
    return llm

def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(NUMBER_OF_CHUNKS_TO_COMBINE)
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(document['chunk_doc'].page_content for document in chunkId_chunkDoc_list[i:i + chunks_to_combine]) for i
        in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)]
    combined_chunks_ids = [[document['chunk_id'] for document in chunkId_chunkDoc_list[i:i + chunks_to_combine]] for i
                           in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(Document(page_content=combined_chunks_page_content[i],
                                                     metadata={"combined_chunk_ids": combined_chunks_ids[i]}))
    return combined_chunk_document_list

# def get_graph_from_OpenAI(model_version, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
#     futures = []
#     graph_document_list = []
#     combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
#     llm = get_llm(model_version)
#     llm_transformer = LLMGraphTransformer(llm=llm,
#                                           node_properties=["description"],
#                                           allowed_nodes=allowedNodes,
#                                           allowed_relationships=allowedRelationship,
#                                           )
#
#     with ThreadPoolExecutor(max_workers=10) as executor:
#         for chunk in combined_chunk_document_list:
#             futures.append(executor.submit(llm_transformer.convert_to_graph_documents, [chunk]))
#
#         for i, future in enumerate(concurrent.futures.as_completed(futures)):
#             try:
#                 graph_document = future.result()
#                 graph_document_list.append(graph_document[0])
#             except:
#                 pass
#     return graph_document_list

def get_graph_document_list(
        llm, combined_chunk_document_list, allowedNodes, allowedRelationship, use_function=True
):
    futures = []
    graph_document_list = []
    if not use_function:
        node_properties = False
    else:
        node_properties = ["description"]
    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=node_properties,
        allowed_nodes=allowedNodes,
        allowed_relationships=allowedRelationship,
        use_function_call=use_function
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        for chunk in combined_chunk_document_list:
            chunk_doc = Document(
                page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata
            )
            futures.append(
                executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc])
            )

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])

    return graph_document_list

def get_graph_from_OpenAI(model_version, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    futures = []
    graph_document_list = []

    combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)

    llm = get_llm(model_version)
    use_function_call = USE_FUNCTION_CALL


    return get_graph_document_list(llm, combined_chunk_document_list, allowedNodes, allowedRelationship,
                                   use_function_call)

def generate_graphDocuments(model: str, graph: Neo4jGraph, chunkId_chunkDoc_list: List, allowedNodes=None,
                            allowedRelationship=None):
    if allowedNodes is None or allowedNodes == "":
        allowedNodes = []
    else:
        allowedNodes = allowedNodes.split(',')
    if allowedRelationship is None or allowedRelationship == "":
        allowedRelationship = []
    else:
        allowedRelationship = allowedRelationship.split(',')

    logging.info(f"allowedNodes: {allowedNodes}, allowedRelationship: {allowedRelationship}")
    graph_documents = []
    if model:
        graph_documents = get_graph_from_OpenAI(MODEL_VERSIONS[model], graph, chunkId_chunkDoc_list, allowedNodes,
                                                allowedRelationship)
        print(graph_documents)
    else:
        raise Exception('Invalid LLM Model')

    logging.info(f"graph_documents = {len(graph_documents)}")
    return graph_documents


def create_graph_database_connection(uri, userName, password):
  graph = Neo4jGraph(url=uri, username=userName, password=password, refresh_schema=False, sanitize=True)
  return graph

class CreateChunksofDocument:
    def __init__(self, pages: list[Document], graph: Neo4jGraph):
        self.pages = pages
        self.graph = graph
    def split_file_into_chunks(self):
        logging.info("Split file into smaller chunks")
        text_splitter = TokenTextSplitter(chunk_size=5000, chunk_overlap=20)
        if 'page' in self.pages[0].metadata:
            chunks = []
            for i, document in enumerate(self.pages):
                page_number = i + 1
                for chunk in text_splitter.split_documents([document]):
                    chunks.append(Document(page_content=chunk.page_content, metadata={'page_number': page_number}))
        else:
            chunks = text_splitter.split_documents(self.pages)
        return chunks

def load_document_content(file_path):
    if Path(file_path).suffix.lower() == '.pdf':
        return PyMuPDFLoader(file_path)
    else:
        return UnstructuredFileLoader(file_path, encoding="utf-8", mode="elements")

def get_pages_with_page_numbers(unstructured_pages):
    pages = []
    page_number = 1
    page_content = ''
    metadata = {}
    for page in unstructured_pages:
        if 'page_number' in page.metadata:
            if page.metadata['page_number'] == page_number:
                page_content += page.page_content
                metadata = {'source': page.metadata['source'], 'page_number': page_number,
                            'filename': page.metadata['filename'],
                            'filetype': page.metadata['filetype'],
                            'total_pages': unstructured_pages[-1].metadata['page_number']}

            if page.metadata['page_number'] > page_number:
                page_number += 1
                if not metadata:
                    metadata = {'total_pages': unstructured_pages[-1].metadata['page_number']}
                pages.append(Document(page_content=page_content, metadata=metadata))
                page_content = ''

            if page == unstructured_pages[-1]:
                if not metadata:
                    metadata = {'total_pages': unstructured_pages[-1].metadata['page_number']}
                pages.append(Document(page_content=page_content, metadata=metadata))

        elif page.metadata['category'] == 'PageBreak' and page != unstructured_pages[0]:
            page_number += 1
            pages.append(Document(page_content=page_content, metadata=metadata))
            page_content = ''
            metadata = {}

        else:
            page_content += page.page_content
            metadata_with_custom_page_number = {'source': page.metadata['source'],
                                                'page_number': 1, 'filename': page.metadata['filename'],
                                                'filetype': page.metadata['filetype'], 'total_pages': 1}
            if page == unstructured_pages[-1]:
                pages.append(Document(page_content=page_content, metadata=metadata_with_custom_page_number))
    return pages

def get_documents_from_file_by_path(file_path,file_name,using_html_pre):
    file_path = Path(file_path)
    if file_path.exists():
        logging.info(f'file {file_name} processing')
        file_extension = file_path.suffix.lower()
        if using_html_pre:
            pages = None
        else:
            try:
                loader = load_document_content(file_path)
                if file_extension == ".pdf":
                    pages = loader.load()
                else:
                    unstructured_pages = loader.load()
                    pages= get_pages_with_page_numbers(unstructured_pages)
            except Exception as e:
                raise Exception('Error while reading the file content or metadata')
    else:
        logging.info(f'File {file_name} does not exist')
        raise Exception(f'File {file_name} does not exist')
    return file_name, pages , file_extension



def update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name):
    isEmbedding = IS_EMBEDDING
    data_for_query = []
    logging.info(f"update embedding and vector index for chunks")
    for row in chunkId_chunkDoc_list:
        if isEmbedding.upper() == "TRUE":
            embeddings_arr = EMBEDDING_FUNCTION.embed_query(row['chunk_doc'].page_content)
            data_for_query.append({
                "chunkId": row['chunk_id'],
                "embeddings": embeddings_arr
            })
            graph.query("""CREATE VECTOR INDEX `vector` if not exists for (c:Chunk) on (c.embedding)
                            OPTIONS {indexConfig: {
                            `vector.dimensions`: $dimensions,
                            `vector.similarity_function`: 'cosine'
                            }}
                        """,
                        {
                            "dimensions": dimension
                        }
                        )

    query_to_create_embedding = """
        UNWIND $data AS row
        MATCH (d:Document {fileName: $fileName})
        MERGE (c:Chunk {id: row.chunkId})
        SET c.embedding = row.embeddings
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_embedding, params={"fileName": file_name, "data": data_for_query})

def get_chunk_and_graphDocument(graph_document_list, chunkId_chunkDoc_list):
    logging.info("creating list of chunks and graph documents in get_chunk_and_graphDocument func")
    lst_chunk_chunkId_document = []
    for graph_document in graph_document_list:
        for chunk_id in graph_document.source.metadata['combined_chunk_ids']:
            lst_chunk_chunkId_document.append({'graph_doc': graph_document, 'chunk_id': chunk_id})

    return lst_chunk_chunkId_document

def create_relation_between_chunks(graph, file_name, chunks: List[Document]) -> list:
    logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
    current_chunk_id = ""
    lst_chunks_including_hash = []
    batch_data = []
    relationships = []
    offset = 0
    for i, chunk in enumerate(chunks):
        page_content_sha1 = hashlib.sha1(chunk.page_content.encode())
        previous_chunk_id = current_chunk_id
        current_chunk_id = page_content_sha1.hexdigest()
        position = i + 1
        if i > 0:
            # offset += len(tiktoken.encoding_for_model("gpt2").encode(chunk.page_content))
            offset += len(chunks[i - 1].page_content)
        if i == 0:
            firstChunk = True
        else:
            firstChunk = False
        metadata = {"position": position, "length": len(chunk.page_content), "content_offset": offset}
        chunk_document = Document(
            page_content=chunk.page_content, metadata=metadata
        )

        chunk_data = {
            "id": current_chunk_id,
            "pg_content": chunk_document.page_content,
            "position": position,
            "length": chunk_document.metadata["length"],
            "f_name": file_name,
            "previous_id": previous_chunk_id,
            "content_offset": offset
        }

        if 'page_number' in chunk.metadata:
            chunk_data['page_number'] = chunk.metadata['page_number']

        if 'start_time' in chunk.metadata and 'end_time' in chunk.metadata:
            chunk_data['start_time'] = chunk.metadata['start_time']
            chunk_data['end_time'] = chunk.metadata['end_time']

        batch_data.append(chunk_data)

        lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk})

        # create relationships between chunks
        if firstChunk:
            relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
        else:
            relationships.append({
                "type": "NEXT_CHUNK",
                "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
                "current_chunk_id": current_chunk_id
            })

    query_to_create_chunk_and_PART_OF_relation = """
        UNWIND $batch_data AS data
        MERGE (c:Chunk {id: data.id})
        SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, c.content_offset=data.content_offset
        WITH data, c
        SET c.page_number = CASE WHEN data.page_number IS NOT NULL THEN data.page_number END,
            c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
            c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END
        WITH data, c
        MATCH (d:Document {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})

    query_to_create_FIRST_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (d:Document {fileName: $f_name})
        MATCH (c:Chunk {id: relationship.chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
    graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})

    query_to_create_NEXT_CHUNK_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (c:Chunk {id: relationship.current_chunk_id})
        WITH c, relationship
        MATCH (pc:Chunk {id: relationship.previous_chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
    graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})

    return lst_chunks_including_hash

def save_graphDocuments_in_neo4j(graph:Neo4jGraph, graph_document_list:List[GraphDocument]):
    graph.add_graph_documents(graph_document_list)

def merge_relationship_between_chunk_and_entites(graph: Neo4jGraph, graph_documents_chunk_chunk_Id: list):
    batch_data = []
    logging.info("Create HAS_ENTITY relationship between chunks and entities")
    chunk_node_id_set = 'id:"{}"'
    for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
        for node in graph_doc_chunk_id['graph_doc'].nodes:
            query_data = {
                'chunk_id': graph_doc_chunk_id['chunk_id'],
                'node_type': node.type,
                'node_id': node.id
            }
            batch_data.append(query_data)

    if batch_data:
        unwind_query = """
                    UNWIND $batch_data AS data
                    MATCH (c:Chunk {id: data.chunk_id})
                    CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                    MERGE (c)-[:HAS_ENTITY]->(n)
                """
        graph.query(unwind_query, params={"batch_data": batch_data})

def processing_chunks(chunks, graph, file_name, model, allowedNodes, allowedRelationship, node_count, rel_count):
    chunkId_chunkDoc_list = create_relation_between_chunks(graph, file_name, chunks)
    # create vector index and update chunk node with embedding
    update_embedding_create_vector_index(graph, chunkId_chunkDoc_list, file_name)
    logging.info("Get graph document list from models")
    graph_documents = generate_graphDocuments(model, graph, chunkId_chunkDoc_list, allowedNodes, allowedRelationship)
    save_graphDocuments_in_neo4j(graph, graph_documents)
    chunks_and_graphDocuments_list = get_chunk_and_graphDocument(graph_documents, chunkId_chunkDoc_list)
    merge_relationship_between_chunk_and_entites(graph, chunks_and_graphDocuments_list)
    # return graph_documents

    distinct_nodes = set()
    relations = []
    for graph_document in graph_documents:
        # get distinct nodes
        for node in graph_document.nodes:
            node_id = node.id
            node_type = node.type
            if (node_id, node_type) not in distinct_nodes:
                distinct_nodes.add((node_id, node_type))
    # get all relations
    for relation in graph_document.relationships:
        relations.append(relation.type)

    node_count += len(distinct_nodes)
    rel_count += len(relations)
    print(f'node count internal func:{node_count}')
    print(f'relation count internal func:{rel_count}')
    return node_count, rel_count

def processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship,is_uploaded_from_local=None,
                      merged_file_path=None, uri=None):
    start_time = datetime.now()
    graphDb_data_Access = graphDBdataAccess(graph)
    result = graphDb_data_Access.get_current_status_document_node(file_name)
    logging.info("Break down file into chunks")
    bad_chars = ['"', "\n", "'"]
    for i in range(0, len(pages)):
        text = pages[i].page_content
        for j in bad_chars:
            if j == '\n':
                text = text.replace(j, ' ')
            else:
                text = text.replace(j, '')
        pages[i] = Document(page_content=str(text), metadata=pages[i].metadata)
    create_chunks_obj = CreateChunksofDocument(pages, graph)
    chunks = create_chunks_obj.split_file_into_chunks()

    if result[0]['Status'] != 'Processing':
        obj_source_node = sourceNode()
        status = "Processing"
        obj_source_node.file_name = file_name
        obj_source_node.status = status
        obj_source_node.total_chunks = len(chunks)
        try:
            obj_source_node.total_pages = len(pages)
        except:
            obj_source_node.total_pages = None
        obj_source_node.model = model
        logging.info(file_name)
        logging.info(obj_source_node)
        graphDb_data_Access.update_source_node(obj_source_node)

        logging.info('Update the status as Processing')
        update_graph_chunk_processed = int(UPDATE_GRAPH_CHUNKS_PROCESSED)
        # selected_chunks = []
        is_cancelled_status = False
        job_status = "Completed"
        node_count = 0
        rel_count = 0
        for i in range(0, len(chunks), update_graph_chunk_processed):
            select_chunks_upto = i + update_graph_chunk_processed
            logging.info(f'Selected Chunks upto: {select_chunks_upto}')
            if len(chunks) <= select_chunks_upto:
                select_chunks_upto = len(chunks)
            selected_chunks = chunks[i:select_chunks_upto]
            result = graphDb_data_Access.get_current_status_document_node(file_name)
            is_cancelled_status = result[0]['is_cancelled']
            logging.info(f"Value of is_cancelled : {result[0]['is_cancelled']}")
            if bool(is_cancelled_status) == True:
                job_status = "Cancelled"
                logging.info('Exit from running loop of processing file')
                exit
            else:
                node_count, rel_count = processing_chunks(selected_chunks, graph, file_name, model, allowedNodes,
                                                          allowedRelationship, node_count, rel_count)
                end_time = datetime.now()
                processed_time = end_time - start_time

                obj_source_node = sourceNode()
                obj_source_node.file_name = file_name
                obj_source_node.updated_at = end_time
                obj_source_node.processing_time = processed_time
                obj_source_node.node_count = node_count
                obj_source_node.processed_chunk = select_chunks_upto
                obj_source_node.relationship_count = rel_count
                graphDb_data_Access.update_source_node(obj_source_node)

        result = graphDb_data_Access.get_current_status_document_node(file_name)
        is_cancelled_status = result[0]['is_cancelled']
        if bool(is_cancelled_status) == True:
            logging.info(f'Is_cancelled True at the end extraction')
            job_status = 'Cancelled'
        logging.info(f'Job Status at the end : {job_status}')
        end_time = datetime.now()
        processed_time = end_time - start_time
        obj_source_node = sourceNode()
        obj_source_node.file_name = file_name
        obj_source_node.status = job_status
        obj_source_node.processing_time = processed_time

        graphDb_data_Access.update_source_node(obj_source_node)
        logging.info('Updated the nodeCount and relCount properties in Docuemnt node')
        logging.info(f'file:{file_name} extraction has been completed')

        return {
            "fileName": file_name,
            "nodeCount": node_count,
            "relationshipCount": rel_count,
            "processingTime": round(processed_time.total_seconds(), 2),
            "status": job_status,
            "model": model,
            "success_count": 1
        }
    else:
        logging.info('File does not process because it\'s already in Processing status')

def extract_graph_from_file_local_file(graph, model, merged_file_path, fileName, allowedNodes, allowedRelationship,uri):

    logging.info(f'Process file name :{fileName}')

    file_name, pages, file_extension = get_documents_from_file_by_path(merged_file_path,fileName,using_html_pre=USING_HTML_PRE)

    if (pages==None or len(pages)==0) and  USING_HTML_PRE != True:
        raise Exception(f'Pdf content is not available for file : {file_name}')

    return processing_source(graph, model, file_name, pages, allowedNodes, allowedRelationship, True, merged_file_path, uri)

class graphDBdataAccess:

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def update_exception_db(self, file_name, exp_msg):
        try:
            job_status = "Failed"
            result = self.get_current_status_document_node(file_name)
            is_cancelled_status = result[0]['is_cancelled']
            if bool(is_cancelled_status) == True:
                job_status = 'Cancelled'
            self.graph.query(
                """MERGE(d:Document {fileName :$fName}) SET d.status = $status, d.errorMessage = $error_msg""",
                {"fName": file_name, "status": job_status, "error_msg": exp_msg})
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error in updating document node status as failed: {error_message}")
            raise Exception(error_message)

    def create_source_node(self, obj_source_node: sourceNode):
        try:
            job_status = "New"
            logging.info("creating source node if does not exist")
            self.graph.query("""MERGE(d:Document {fileName :$fn}) SET d.fileSize = $fs, d.fileType = $ft ,
                            d.status = $st, d.url = $url, d.awsAccessKeyId = $awsacc_key_id, 
                            d.fileSource = $f_source, d.createdAt = $c_at, d.updatedAt = $u_at, 
                            d.processingTime = $pt, d.errorMessage = $e_message, d.nodeCount= $n_count, 
                            d.relationshipCount = $r_count, d.model= $model, d.gcsBucket=$gcs_bucket, 
                            d.gcsBucketFolder= $gcs_bucket_folder, d.language= $language,d.gcsProjectId= $gcs_project_id,
                            d.is_cancelled=False, d.total_chunks=0, d.processed_chunk=0, d.total_pages=$total_pages,
                            d.access_token=$access_token""",
                             {"fn": obj_source_node.file_name, "fs": obj_source_node.file_size,
                              "ft": obj_source_node.file_type, "st": job_status,
                              "url": obj_source_node.url,
                              "awsacc_key_id": obj_source_node.awsAccessKeyId, "f_source": obj_source_node.file_source,
                              "c_at": obj_source_node.created_at,
                              "u_at": obj_source_node.created_at, "pt": 0, "e_message": '', "n_count": 0, "r_count": 0,
                              "model": obj_source_node.model,
                              "gcs_bucket": obj_source_node.gcsBucket,
                              "gcs_bucket_folder": obj_source_node.gcsBucketFolder,
                              "language": obj_source_node.language, "gcs_project_id": obj_source_node.gcsProjectId,
                              "total_pages": obj_source_node.total_pages,
                              "access_token": obj_source_node.access_token})
        except Exception as e:
            error_message = str(e)
            logging.info(f"error_message = {error_message}")
            self.update_exception_db(self, obj_source_node.file_name, error_message)
            raise Exception(error_message)

    def update_source_node(self, obj_source_node: sourceNode):
        try:

            params = {}
            if obj_source_node.file_name is not None and obj_source_node.file_name != '':
                params['fileName'] = obj_source_node.file_name

            if obj_source_node.status is not None and obj_source_node.status != '':
                params['status'] = obj_source_node.status

            if obj_source_node.created_at is not None:
                params['createdAt'] = obj_source_node.created_at

            if obj_source_node.updated_at is not None:
                params['updatedAt'] = obj_source_node.updated_at

            if obj_source_node.processing_time is not None and obj_source_node.processing_time != 0:
                params['processingTime'] = round(obj_source_node.processing_time.total_seconds(), 2)

            if obj_source_node.node_count is not None and obj_source_node.node_count != 0:
                params['nodeCount'] = obj_source_node.node_count

            if obj_source_node.relationship_count is not None and obj_source_node.relationship_count != 0:
                params['relationshipCount'] = obj_source_node.relationship_count

            if obj_source_node.model is not None and obj_source_node.model != '':
                params['model'] = obj_source_node.model

            if obj_source_node.total_pages is not None and obj_source_node.total_pages != 0:
                params['total_pages'] = obj_source_node.total_pages

            if obj_source_node.total_chunks is not None and obj_source_node.total_chunks != 0:
                params['total_chunks'] = obj_source_node.total_chunks

            if obj_source_node.is_cancelled is not None and obj_source_node.is_cancelled != False:
                params['is_cancelled'] = obj_source_node.is_cancelled

            if obj_source_node.processed_chunk is not None and obj_source_node.processed_chunk != 0:
                params['processed_chunk'] = obj_source_node.processed_chunk

            param = {"props": params}
            print(f'Base Param value 1 : {param}')
            query = "MERGE(d:Document {fileName :$props.fileName}) SET d += $props"
            logging.info("Update source node properties")
            self.graph.query(query, param)
        except Exception as e:
            # error_message = str(e)
            # self.update_exception_db(self.file_name, error_message)
            # raise Exception(error_message)
            error_message = str(e)
            # Fix 2: use obj_source_node.file_name instead of self.file_name.
            # Ensure file_name exists and provide a default when it does not.
            f_name = obj_source_node.file_name if obj_source_node.file_name else "Unknown"
            self.update_exception_db(f_name, error_message)
            raise Exception(error_message)

    def get_source_list(self):
        """
        Args:
            uri: URI of the graph to extract
            db_name: db_name is database name to connect to graph db
            userName: Username to use for graph creation ( if None will use username from config file )
            password: Password to use for graph creation ( if None will use password from config file )
            file: File object containing the PDF file to be used
            model: Type of model to use ('Diffbot'or'OpenAI GPT')
        Returns:
        Returns a list of sources that are in the database by querying the graph and
        sorting the list by the last updated date.
        """

        logging.info("Get existing files list from graph")
        query = "MATCH(d:Document) WHERE d.fileName IS NOT NULL RETURN d ORDER BY d.updatedAt DESC"
        result = self.graph.query(query)
        list_of_json_objects = [entry['d'] for entry in result]
        return list_of_json_objects

    def update_KNN_graph(self):
        """
        Update the graph node with SIMILAR relationship where embedding scrore match
        """
        index = self.graph.query("""show indexes yield * where type = 'VECTOR' and name = 'vector'""")

        knn_min_score = KNN_MIN_SCORE
        if len(index) > 0:
            logging.info('update KNN graph')
            self.graph.query("""MATCH (c:Chunk)
                                    WHERE c.embedding IS NOT NULL AND count { (c)-[:SIMILAR]-() } < 5
                                    CALL db.index.vector.queryNodes('vector', 6, c.embedding) yield node, score
                                    WHERE node <> c and score >= $score MERGE (c)-[rel:SIMILAR]-(node) SET rel.score = score
                                """,
                             {"score": float(knn_min_score)}
                             )
        else:
            logging.info("Vector index does not exist, So KNN graph not update")

    def connection_check(self):
        """
        Args:
            uri: URI of the graph to extract
            userName: Username to use for graph creation ( if None will use username from config file )
            password: Password to use for graph creation ( if None will use password from config file )
            db_name: db_name is database name to connect to graph db
        Returns:
        Returns a status of connection from NEO4j is success or failure
        """
        if self.graph:
            return "Connection Successful"

    def execute_query(self, query, param=None):
        return self.graph.query(query, param)

    def get_current_status_document_node(self, file_name):
        query = """
                MATCH(d:Document {fileName : $file_name}) RETURN d.status AS Status , d.processingTime AS processingTime, 
                d.nodeCount AS nodeCount, d.model as model, d.relationshipCount as relationshipCount,
                d.total_pages AS total_pages, d.total_chunks AS total_chunks , d.fileSize as fileSize, 
                d.is_cancelled as is_cancelled, d.processed_chunk as processed_chunk, d.fileSource as fileSource
                """
        param = {"file_name": file_name}
        return self.execute_query(query, param)

    # def delete_file_from_graph(self, filenames, source_types, deleteEntities: str, merged_dir: str, uri):
    #     # filename_list = filenames.split(',')
    #     filename_list = list(map(str.strip, json.loads(filenames)))
    #     source_types_list = list(map(str.strip, json.loads(source_types)))
    #     gcs_file_cache = os.environ.get('GCS_FILE_CACHE')
    #     # source_types_list = source_types.split(',')
    #     for (file_name, source_type) in zip(filename_list, source_types_list):
    #         merged_file_path = os.path.join(merged_dir, file_name)
    #         if source_type == 'local file' :
    #             logging.info(f'Deleted File Path: {merged_file_path} and Deleted File Name : {file_name}')
    #             delete_uploaded_local_file(merged_file_path, file_name)
    #     query_to_delete_document = """
    #        MATCH (d:Document) where d.fileName in $filename_list and d.fileSource in $source_types_list
    #         with collect(d) as documents
    #         unwind documents as d
    #         optional match (d)<-[:PART_OF]-(c:Chunk)
    #         detach delete c, d
    #         return count(*) as deletedChunks
    #         """
    #
    #     MATCH(d: Document) where d.fileName in ["en202303.pdf"]
    #     RETURN d LIMIT 25

        # query_to_delete_document_and_entities = """
        #     MATCH (d:Document) where d.fileName in $filename_list and d.fileSource in $source_types_list
        #     with collect(d) as documents
        #     unwind documents as d
        #     optional match (d)<-[:PART_OF]-(c:Chunk)
        #     // if delete-entities checkbox is set
        #     call { with  c, documents
        #         match (c)-[:HAS_ENTITY]->(e)
        #         // belongs to another document
        #         where not exists {  (d2)<-[:PART_OF]-()-[:HAS_ENTITY]->(e) WHERE NOT d2 IN documents }
        #         detach delete e
        #         return count(*) as entities
        #     }
        #     detach delete c, d
        #     return sum(entities) as deletedEntities, count(*) as deletedChunks
        #     """
        # param = {"filename_list": filename_list, "source_types_list": source_types_list}
        # if deleteEntities == "true":
        #     result = self.execute_query(query_to_delete_document_and_entities, param)
        #     logging.info(
        #         f"Deleting {len(filename_list)} documents = '{filename_list}' from '{source_types_list}' from database")
        # else:
        #     result = self.execute_query(query_to_delete_document, param)
        #     logging.info(
        #         f"Deleting {len(filename_list)} documents = '{filename_list}' from '{source_types_list}' with their entities from database")
        #
        # return result, len(filename_list)

def test_graph_from_file_local_file(file_name):
    graph = create_graph_database_connection(uri, userName, password)
    try:
        file_name = file_name
        obj_source_node = sourceNode()
        obj_source_node.file_name = file_name
        obj_source_node.file_type = 'txt'
        obj_source_node.file_size = '10'
        obj_source_node.file_source = 'local file'
        obj_source_node.model = model
        obj_source_node.created_at = datetime.now()
        graphDb_data_Access = graphDBdataAccess(graph)
        graphDb_data_Access.create_source_node(obj_source_node)
        merged_file_path =  os.path.join(MERGED_DIR,file_name)

        local_file_result = extract_graph_from_file_local_file(graph, model, merged_file_path,file_name, '', '',"")
        print(local_file_result)
        logging.info(f"Info: {local_file_result}")
        return {
            "status": "success",
            "file_name": file_name,
            "info": local_file_result
        }
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error processing {file_name}: {error_msg}")

        try:
            graphDb_data_Access = graphDBdataAccess(graph)
            graphDb_data_Access.update_exception_db(file_name, error_msg)
        except:
            pass

        return {
            "status": "failed",
            "file_name": file_name,
            "error": error_msg
        }
    finally:
        try:
            graph._driver.close()
        except:
            pass



def update_graph(graph):
  """
  Update the graph node with SIMILAR relationship where embedding score match
  """
  graph_DB_dataAccess = graphDBdataAccess(graph)
  graph_DB_dataAccess.update_KNN_graph()


def get_neo4j_retriever(graph,retrieval_query, EMBEDDING_FUNCTION,index_name="vector",search_k=CHAT_SEARCH_KWARG_K,score_threshold=CHAT_SEARCH_KWARG_SCORE_THRESHOLD):

    neo_db = Neo4jVector.from_existing_index(
        embedding=EMBEDDING_FUNCTION,
        index_name=index_name,
        retrieval_query=retrieval_query,
        graph=graph
    )
    retriever = neo_db.as_retriever(search_kwargs={'k': search_k, "score_threshold": score_threshold})
    return retriever

def create_document_retriever_chain(llm,retriever,EMBEDDING_FUNCTION):
    question_template= "Given the below conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else."

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_template),
            MessagesPlaceholder(variable_name="messages")
        ])
    splitter = TokenTextSplitter(chunk_size=CHAT_DOC_SPLIT_SIZE, chunk_overlap=0)
    embeddings_filter = EmbeddingsFilter(embeddings=EMBEDDING_FUNCTION, similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD)

    output_parser = StrOutputParser()

    # redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter, embeddings_filter])

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retriever
    )
    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | compression_retriever,
        ),
        query_transform_prompt | llm | output_parser | compression_retriever,
    ).with_config(run_name="chat_retriever_chain")

    return query_transforming_retriever_chain


def format_documents(documents):
    # prompt_token_cutoff = 4

    sorted_documents = sorted(documents, key=lambda doc: doc.state["query_similarity_score"], reverse=True)
    # sorted_documents = sorted_documents[:prompt_token_cutoff]

    formatted_docs = []
    sources = set()
    chunkdetails= []
    for doc in sorted_documents:
        source = doc.metadata['source']
        chunkdetail = doc.metadata['chunkdetails']

        sources.add(source)
        chunkdetails.append(chunkdetail)

        formatted_doc = (
            "Document start\n"
            f"This Document belongs to the source {source}\n"
            f"Content: {doc.page_content}\n"
            "Document end\n"
        )
        formatted_docs.append(formatted_doc)

    return "\n\n".join(formatted_docs), sources, chunkdetails

def get_relevant_assay(query_content,graph,EMBEDDING_FUNCTION):
    llm = get_llm(MODEL_VERSIONS[model])
    retriever = get_neo4j_retriever(graph=graph,EMBEDDING_FUNCTION=EMBEDDING_FUNCTION, retrieval_query=VECTOR_GRAPH_SEARCH_QUERY,)
    doc_retriever = create_document_retriever_chain(llm, retriever,EMBEDDING_FUNCTION=EMBEDDING_FUNCTION)
    messages = HumanMessage(content=query_content)
    docs = doc_retriever.invoke({
        "messages": [messages]})

    return docs

class Neo4jConnection:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    def close(self):
        self.driver.close()
    def run_query(self, query, parameters):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

def get_chunk_rel_aids(conn,chunk_id,conn_counts=5):

    query = """
    MATCH (startChunk:Chunk {id: $chunk_id})
    MATCH (startChunk)-[:HAS_ENTITY]->(entity)<-[:HAS_ENTITY]-(chunk)-[:PART_OF]->(doc:Document)
    RETURN doc
    """
    results = conn.run_query(query,{"chunk_id": chunk_id})
    conn.close()
    get_chunk_rel_assays=[]
    for result in results:
        get_chunk_rel_assays.append(result["doc"]['fileName'])
    assay_count = pd.DataFrame(get_chunk_rel_assays, columns=['assay_aid'])
    file_name_counts = assay_count['assay_aid'].value_counts()
    frequent_files = file_name_counts[file_name_counts >= conn_counts]
    return  frequent_files

def init_worker():
    global EMBEDDING_FUNCTION, dimension
    EMBEDDING_FUNCTION, dimension = load_embedding_model(EMBEDDING_MODEL)

# if __name__ == '__main__':

#     # graph = create_graph_database_connection(uri, userName, password)
#     # EMBEDDING_FUNCTION, dimension = load_embedding_model(EMBEDDING_MODEL)
#     # update_graph(graph)
#     #
#     # file_list = os.listdir(MERGED_DIR)

#     # print(f"Processing {len(file_list)} files with multiprocessing...")
#     #
#     # pool = Pool(3, initializer=init_worker)

#     # pool.map(test_graph_from_file_local_file, file_list[468:])
#     #
#     # pool.close()
#     # pool.join()

#     main_graph = create_graph_database_connection(uri, userName, password)
#     EMBEDDING_FUNCTION, dimension = load_embedding_model(EMBEDDING_MODEL)
#     update_graph(main_graph)
#     main_graph._driver.close() 

#     error_files= pd.read_csv("./bioassays_without_successful_kg_extraction.csv")
#     file_list = error_files["FileName"].tolist()

#     # Define the error log path.
#     error_log_file = "processing_errors_2.txt"

#     # Clear or create the error log file.
#     with open(error_log_file, "w", encoding="utf-8") as f:
#         f.write(f"Start processing at {datetime.now()}\n")

#     print(f"Processing {len(file_list)} files...")

#     # Use initializer=init_worker.
#     pool = Pool(3, initializer=init_worker)

#     # Use imap_unordered to receive results as they become available.
#     results_iterator = pool.imap_unordered(test_graph_from_file_local_file, file_list)

#     # Process results as they arrive.
#     for result in tqdm(results_iterator, total=len(file_list)):
#         if result['status'] == 'failed':
#             file_name = result['file_name']
#             error_msg = result['error']

#             print(f"Failed: {file_name}")
#             with open(error_log_file, "a", encoding="utf-8") as f:
#                 f.write(f"{file_name} | Error: {error_msg}\n")
#         else:
#             pass

#     pool.close()
#     pool.join()
#     print("All tasks finished.")

