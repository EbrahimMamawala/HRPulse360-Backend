import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = "PerformanceData"
CSV_FILE = "hr_synthetic.csv"

df = pd.read_csv(CSV_FILE)

client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})
print("Cleared existing PerformanceData collection")

PERFORMANCE_WEIGHTS = {
    'performance_rating': 0.30,
    'engagement_score': 0.20,
    'courses_completed': 0.10,
    'tenure_score': 0.15,
    'salary_hike': 0.10,
    'bonus_percent': 0.10,
    'overtime_penalty': 0.05,
}

def parse_date(date_str):
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return datetime.strptime(str(date_str), '%Y-%m')
    except:
        return None

def normalize_score(series, min_val=None, max_val=None):
    if min_val is None:
        min_val = series.min()
    if max_val is None:
        max_val = series.max()
    if max_val == min_val:
        return pd.Series([50] * len(series), index=series.index)
    return ((series - min_val) / (max_val - min_val)) * 100

def calculate_tenure_score(years):
    if years < 1:
        return 40
    elif years < 3:
        return 60 + (years * 10)
    elif years <= 7:
        return 90 + ((7 - years) * 2)
    elif years <= 10:
        return 85 - ((years - 7) * 5)
    else:
        return 70

def calculate_overtime_penalty(overtime_done, avg_hours):
    if overtime_done == 'No':
        return 100
    else:
        if avg_hours <= 170:
            return 90
        elif avg_hours <= 190:
            return 75
        elif avg_hours <= 210:
            return 60
        else:
            return 40

def calculate_performance_scores(df, target_date):
    month_start = target_date.replace(day=1)
    next_month = month_start + relativedelta(months=1)
    df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
    df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)

    active_employees = df[
        (df['JoiningDate'] < next_month) &
        ((df['TerminationDate'].isna()) | (df['TerminationDate'] >= month_start))
    ].copy()

    active_employees = active_employees[active_employees['Attrition'] == 'No'].copy()
    if len(active_employees) == 0:
        return None

    active_employees['perf_rating_score'] = normalize_score(active_employees['PerformanceRating'], 0, 5)
    active_employees['engagement_score_normalized'] = normalize_score(active_employees['EngagementScore'], 0, 5)

    max_courses = active_employees['CountCoursesDoneLastYear'].max()
    if max_courses > 0:
        active_employees['courses_score'] = (active_employees['CountCoursesDoneLastYear'] / max_courses) * 100
    else:
        active_employees['courses_score'] = 50

    active_employees['tenure_score'] = active_employees['YearsAtCompany'].apply(calculate_tenure_score)
    active_employees['salary_hike_score'] = normalize_score(active_employees['PercentSalaryHike'])
    active_employees['bonus_score'] = normalize_score(active_employees['BonusPercent'])

    active_employees['overtime_score'] = active_employees.apply(
        lambda row: calculate_overtime_penalty(row['OvertimeDone'], row['AvgHoursWorkedPerMonth']), axis=1
    )

    active_employees['overall_performance_score'] = (
        active_employees['perf_rating_score'] * PERFORMANCE_WEIGHTS['performance_rating'] +
        active_employees['engagement_score_normalized'] * PERFORMANCE_WEIGHTS['engagement_score'] +
        active_employees['courses_score'] * PERFORMANCE_WEIGHTS['courses_completed'] +
        active_employees['tenure_score'] * PERFORMANCE_WEIGHTS['tenure_score'] +
        active_employees['salary_hike_score'] * PERFORMANCE_WEIGHTS['salary_hike'] +
        active_employees['bonus_score'] * PERFORMANCE_WEIGHTS['bonus_percent'] +
        active_employees['overtime_score'] * PERFORMANCE_WEIGHTS['overtime_penalty']
    )

    active_employees['overall_performance_score'] = active_employees['overall_performance_score'].round(2)

    def get_performance_tier(score):
        if score >= 85:
            return "Exceptional"
        elif score >= 70:
            return "High Performer"
        elif score >= 55:
            return "Solid Performer"
        elif score >= 40:
            return "Needs Improvement"
        else:
            return "Critical"

    active_employees['performance_tier'] = active_employees['overall_performance_score'].apply(get_performance_tier)
    return active_employees

def get_top_performers_by_department(df_with_scores, top_n=10):
    top_performers_by_dept = []
    for dept in df_with_scores['Department'].unique():
        dept_df = df_with_scores[df_with_scores['Department'] == dept].copy()
        dept_df = dept_df.sort_values('overall_performance_score', ascending=False)
        top_dept = dept_df.head(top_n)

        employees = []
        for idx, (_, row) in enumerate(top_dept.iterrows(), 1):
            employees.append({
                'rank': idx,
                'employeeId': int(row['EmployeeID']),
                'fullName': row['FullName'],
                'jobRole': row['JobRole'],
                'department': dept,
                'overallScore': float(row['overall_performance_score']),
                'performanceTier': row['performance_tier']
            })

        top_performers_by_dept.append({
            'department': dept,
            'topPerformers': employees
        })
    return top_performers_by_dept

def generate_top_performer_data(df, target_date=None):
    if target_date is None:
        target_date = datetime.now()

    print(f"\nGenerating top performer data for {target_date.strftime('%Y-%m')}")

    df_with_scores = calculate_performance_scores(df, target_date)
    if df_with_scores is None or len(df_with_scores) == 0:
        print("No active employees found.")
        return None

    top_performers = get_top_performers_by_department(df_with_scores, top_n=10)
    document = {
        'period': target_date.strftime('%Y-%m'),
        'timestamp': datetime.utcnow(),
        'departmentTopPerformers': top_performers
    }

    collection.replace_one(
        {'period': target_date.strftime('%Y-%m')},
        document,
        upsert=True
    )

    print(f"Inserted top performers per department for {target_date.strftime('%Y-%m')}")
    return document

if __name__ == "__main__":
    try:
        print("Starting Top Performer Data Generation...\n")
        df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
        df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)
        target_date = datetime(2024, 11, 1)
        generate_top_performer_data(df, target_date)
    except Exception as e:
        print("Error:", str(e))
        import traceback
        traceback.print_exc()
    finally:
        client.close()
        print("MongoDB connection closed.")
