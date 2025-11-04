import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = "DashboardData"
CSV_FILE = "hr_synthetic.csv"

df = pd.read_csv(CSV_FILE)

client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})

def parse_monthyear(s):
    return pd.to_datetime(s + "-01", errors="coerce")

df["JoiningMonthYear_dt"] = parse_monthyear(df["JoiningMonthYear"].astype(str))
df["TerminationMonthYear_dt"] = parse_monthyear(df["TerminationMonthYear"].astype(str))

start_date = df["JoiningMonthYear_dt"].min().replace(day=1)
end_date_candidates = pd.concat([df["JoiningMonthYear_dt"], df["TerminationMonthYear_dt"].dropna()])
end_date = end_date_candidates.max().replace(day=1)
months = pd.date_range(start=start_date, end=end_date, freq="MS")

def active_on(df, target_month):
    jm = df["JoiningMonthYear_dt"]
    tm = df["TerminationMonthYear_dt"]
    return df[(jm <= target_month) & ((tm.isna()) | (tm > target_month))]

def new_hires_in_month(df, target_month):
    return df[df["JoiningMonthYear_dt"] == target_month]

def departures_in_month(df, target_month):
    tm = df["TerminationMonthYear_dt"]
    return df[(~tm.isna()) & (tm == target_month)]

monthly_data = []

for m in months:
    total_employees = len(active_on(df, m))
    new_hires = len(new_hires_in_month(df, m))
    departures = len(departures_in_month(df, m))
    monthly_data.append({
        "month": m.strftime("%Y-%m"),
        "total_employees": total_employees,
        "new_hires": new_hires,
        "departures": departures
    })

latest_month = months[-1]
curr_active = active_on(df, latest_month)
prev_active = active_on(df, latest_month - relativedelta(months=1))
same_month_last_year = latest_month - relativedelta(years=1)
last_year_active = active_on(df, same_month_last_year)

total_employees = len(curr_active)
prev_total = len(prev_active)
total_change_pct = round(((total_employees - prev_total) / prev_total) * 100, 2) if prev_total > 0 else 0

terminations_curr = len(departures_in_month(df, latest_month))
terminations_prev = len(departures_in_month(df, latest_month - relativedelta(months=1)))
attrition_rate = round((terminations_curr / prev_total) * 100, 2) if prev_total > 0 else 0
attrition_change = round(attrition_rate - ((terminations_prev / prev_total) * 100), 2) if prev_total > 0 else 0

genders = curr_active["Gender"].fillna("Unknown").str.lower()
male_count = int((genders == "male").sum())
female_count = int((genders == "female").sum())
gender_ratio = f"{round(male_count/female_count, 1) if female_count else male_count}:1"

avg_tenure = round(curr_active["YearsAtCompany"].mean(), 2) if len(curr_active) > 0 else 0
avg_tenure_last_year = round(last_year_active["YearsAtCompany"].mean(), 2) if len(last_year_active) > 0 else 0
tenure_change = round(avg_tenure - avg_tenure_last_year, 2)

dashboard_data = {
    "generated_at": datetime.utcnow(),
    "current_month": latest_month.strftime("%Y-%m"),
    "cards": {
        "total_employees": total_employees,
        "total_change_pct": total_change_pct,
        "attrition_rate": attrition_rate,
        "attrition_change_pct": attrition_change,
        "gender_ratio": gender_ratio,
        "avg_tenure": avg_tenure,
        "avg_tenure_change": tenure_change
    },
    "overview_chart": monthly_data
}

collection.insert_one(dashboard_data)

print("Inserted document successfully:")
print(dashboard_data)
