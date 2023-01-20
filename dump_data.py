import pymongo
import pandas as pd
import json

from flight_fare.config import mongo_client

DATA_FILE_PATH="archive/Data_Train.xlsx"
DATABASE_NAME="ffp"
COLLECTION_NAME="flight_fare"


if __name__=="__main__":
    df = pd.read_excel(DATA_FILE_PATH)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])
    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)