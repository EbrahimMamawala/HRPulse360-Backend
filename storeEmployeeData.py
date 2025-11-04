import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = "EmployeeData"
CSV_FILE = "hr_synthetic.csv"

df = pd.read_csv(CSV_FILE)

essential_cols = [
    "EmployeeID",
    "FullName",
    "Age",
    "Gender",
    "Department",
    "JobRole",
    "JoiningMonthYear",
    "TerminationMonthYear",
    "Attrition"
]
df_essential = df[essential_cols]

records = df_essential.to_dict(orient="records")

client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

delete_result = collection.delete_many({})
print(f"Deleted {delete_result.deleted_count} existing records from {DB_NAME}.{COLLECTION_NAME}")

if records:
    collection.insert_many(records)
    print(f"Inserted {len(records)} employee records into {DB_NAME}.{COLLECTION_NAME}")
else:
    print("No records found in the CSV file.")

client.close()
