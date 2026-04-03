import json
import pandas as pd
import config

def load_PAMS_corpus():

    excel_file_path = config.Dataset_Url
    df = pd.read_excel(excel_file_path)

    # 工艺列表
    with open(config.PAMs_Url, 'r', encoding='utf-8') as f:
        loaded_processes_dict = json.load(f)
        loaded_processes = list(loaded_processes_dict.values())

    return df, loaded_processes

