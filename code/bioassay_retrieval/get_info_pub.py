import requests
from multiprocessing.dummy import Pool
from pathlib import Path
import pandas as pd


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.50"
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


class Get_Data_Pub():

    def __init__(self, aid_list, task_name, output_dir=None):
        self.aid_list = aid_list
        self.task_name = task_name
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR / "bioassay_raw" / task_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.aid_file_path_list = []
        self.cid_failed_url = []

    def get_assay_data(self, dic, check=True):
        url = dic["url"]
        response = requests.get(url=url, headers=headers)
        data = response.content
        if response.status_code == 200:
            out_path = self.output_dir / f"{dic['name']}.csv"
            with open(out_path, "wb") as fp:
                fp.write(data)
                self.aid_file_path_list.append(str(out_path))
        else:
            if check:
                self.cid_failed_url.append(dic)
            print(dic["name"] + ": download failed") 

    def get_aid_csv(self):

        assay_url = []
        for aid in self.aid_list:
            url = f"https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&actvty=all&response_type=save&aid={str(aid)}"
            dic = {"url": url, "name": str(aid)}
            assay_url.append(dic)

        pool = Pool(5)
        pool.map(self.get_assay_data, assay_url)

    def process_assay_data(self):
        for aid in self.aid_list:
            # with open(self.output_dir / f"{aid}.csv", "rb") as fp:
            #     assay_data = fp.read()
            assay_csv =pd.read_csv(self.output_dir / f"{aid}.csv")

            pass


# Example
# get_data = Get_Data_Pub(aid_list=[1195],task_name="neurotoxicity")
# get_data.get_aid_csv()
# get_data.process_assay_data()
