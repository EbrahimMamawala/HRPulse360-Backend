import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = "AttritionData"
CSV_FILE = "hr_synthetic.csv"
MODEL_FILE = "best_pipeline.joblib"

# Load data and model
df = pd.read_csv(CSV_FILE)
model_dict = joblib.load(MODEL_FILE)
model = model_dict['pipeline']

# MongoDB connection
client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Clear existing data
collection.delete_many({})
print("Cleared existing AttritionData collection")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def engineer_features(df):
    """Create engineered features required by the model"""
    df = df.copy()
    df['tenure_ratio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    df['pay_per_experience'] = df['BasePay'] / (df['WorkExperienceYears'] + 1)
    df['ManagerID_freq'] = df.groupby('ManagerID')['ManagerID'].transform('count')
    df['overwork_index'] = df['AvgHoursWorkedPerMonth'] / 160
    return df


def parse_date(date_str):
    """Parse date string in YYYY-MM format"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return datetime.strptime(str(date_str), '%Y-%m')
    except:
        return None


def calculate_percentage_change(current, previous):
    """Calculate percentage change"""
    if previous == 0:
        return 0 if current == 0 else 100
    return ((current - previous) / previous) * 100


# ============================================================================
# MONTHLY METRICS FUNCTIONS
# ============================================================================

def get_month_data(df, target_date):
    """Get metrics for a specific month"""
    month_start = target_date.replace(day=1)
    next_month = month_start + relativedelta(months=1)
    
    # Parse dates
    df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
    df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)
    
    # Active employees at end of month
    active_employees = df[
        (df['JoiningDate'] < next_month) & 
        ((df['TerminationDate'].isna()) | (df['TerminationDate'] >= month_start))
    ].copy()
    
    # Exits this month
    exits_this_month = df[
        (df['TerminationDate'] >= month_start) & 
        (df['TerminationDate'] < next_month)
    ]
    
    # New hires this month
    new_hires_this_month = df[
        (df['JoiningDate'] >= month_start) & 
        (df['JoiningDate'] < next_month)
    ]
    
    return active_employees, exits_this_month, new_hires_this_month


# ============================================================================
# DEPARTMENT METRICS
# ============================================================================

def get_attrition_by_department(df, current_date):
    """Calculate attrition metrics per department"""
    current_month_start = current_date.replace(day=1)
    year_start = current_date.replace(month=1, day=1)
    next_month = current_month_start + relativedelta(months=1)
    
    df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
    df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)
    
    dept_metrics = []
    
    for dept in df['Department'].unique():
        dept_df = df[df['Department'] == dept]
        
        # Active employees at start of year
        active_start = dept_df[
            (dept_df['JoiningDate'] < year_start) & 
            ((dept_df['TerminationDate'].isna()) | (dept_df['TerminationDate'] >= year_start))
        ]
        
        # Exits year to date
        exits_ytd = dept_df[
            (dept_df['TerminationDate'] >= year_start) & 
            (dept_df['TerminationDate'] < next_month)
        ]
        
        # Current active employees
        active_current = dept_df[
            (dept_df['JoiningDate'] < next_month) & 
            ((dept_df['TerminationDate'].isna()) | (dept_df['TerminationDate'] >= current_month_start))
        ]
        
        avg_employees = (len(active_start) + len(active_current)) / 2
        attrition_rate = (len(exits_ytd) / avg_employees * 100) if avg_employees > 0 else 0
        
        dept_metrics.append({
            'department': dept,
            'totalEmployees': len(active_current),
            'exitsYTD': len(exits_ytd),
            'attritionRate': round(attrition_rate, 2)
        })
    
    return dept_metrics


# ============================================================================
# TOP RISK EMPLOYEES
# ============================================================================

def get_top_risk_employees_by_department(df, probabilities):
    """Get top 10 at-risk employees per department"""
    df_risk = df.copy()
    df_risk['AttritionProbability'] = probabilities[:, 1]
    
    # Filter only active employees
    df_risk = df_risk[df_risk['Attrition'] == 'No'].copy()
    
    top_risk_by_dept = []
    
    for dept in df_risk['Department'].unique():
        dept_df = df_risk[df_risk['Department'] == dept].copy()
        dept_df = dept_df.nlargest(10, 'AttritionProbability')
        
        employees = []
        for _, row in dept_df.iterrows():
            employees.append({
                'employeeId': int(row['EmployeeID']),
                'fullName': row['FullName'],
                'age': int(row['Age']),
                'jobRole': row['JobRole'],
                'yearsAtCompany': int(row['YearsAtCompany']),
                'engagementScore': float(row['EngagementScore']),
                'performanceRating': float(row['PerformanceRating']),
                'attritionProbability': round(float(row['AttritionProbability']) * 100, 2)
            })
        
        top_risk_by_dept.append({
            'department': dept,
            'topRiskEmployees': employees
        })
    
    return top_risk_by_dept


# ============================================================================
# ATTRITION TIMELINE
# ============================================================================

def predict_future_attrition(df, current_date):
    """Predict attrition using historical trends - current year only"""
    predictions = []
    
    # Get current year boundaries
    year_start = current_date.replace(month=1, day=1)
    current_month_start = current_date.replace(day=1)
    
    df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
    df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)
    
    # Calculate historical monthly attrition (January to current month)
    month = year_start
    while month <= current_month_start:
        next_month = month + relativedelta(months=1)
        
        active_start = df[
            (df['JoiningDate'] < month) & 
            ((df['TerminationDate'].isna()) | (df['TerminationDate'] >= month))
        ]
        
        exits = df[
            (df['TerminationDate'] >= month) & 
            (df['TerminationDate'] < next_month)
        ]
        
        attrition_rate = (len(exits) / len(active_start) * 100) if len(active_start) > 0 else 0
        
        predictions.append({
            'month': month.strftime('%Y-%m'),
            'type': 'actual',
            'attritionCount': len(exits),
            'attritionRate': round(attrition_rate, 2),
            'totalEmployees': len(active_start)
        })
        
        month = next_month
    
    # Get current active employees
    active_current = df[
        (df['JoiningDate'] < current_month_start + relativedelta(months=1)) & 
        ((df['TerminationDate'].isna()) | (df['TerminationDate'] >= current_month_start))
    ]
    
    # Calculate statistics
    historical_rates = [p['attritionRate'] for p in predictions]
    avg_rate = np.mean(historical_rates)
    std_rate = np.std(historical_rates)
    
    # Calculate trend (simple linear regression on last 3 months)
    recent_rates = historical_rates[-3:] if len(historical_rates) >= 3 else historical_rates
    if len(recent_rates) >= 2:
        trend = (recent_rates[-1] - recent_rates[0]) / len(recent_rates)
    else:
        trend = 0
    
    print(f"\nHistorical Analysis:")
    print(f"  Average Rate: {avg_rate:.2f}%")
    print(f"  Std Deviation: {std_rate:.2f}%")
    print(f"  Recent Trend: {trend:+.2f}% per month")
    
    # Predict remaining months
    month = current_month_start + relativedelta(months=1)
    month_offset = 1
    
    while month.year == current_date.year:
        # Trend-based prediction with dampening
        predicted_rate = avg_rate + (trend * month_offset * 0.5)  # 50% dampening
        predicted_rate = max(0, min(predicted_rate, avg_rate * 2))  # Cap at 2x average
        
        predicted_exits = (predicted_rate / 100) * len(active_current)
        
        # Confidence bounds
        lower_bound = max(0, predicted_rate - 1.5 * std_rate)
        upper_bound = predicted_rate + 1.5 * std_rate
        
        predictions.append({
            'month': month.strftime('%Y-%m'),
            'type': 'predicted',
            'attritionCount': round(predicted_exits),
            'attritionRate': round(predicted_rate, 2),
            'lowerBound': round(lower_bound, 2),
            'upperBound': round(upper_bound, 2),
            'totalEmployees': len(active_current)
        })
        
        month = month + relativedelta(months=1)
        month_offset += 1
    
    return predictions

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_and_store(df, model, current_date=None):
    """Main function to analyze data and store in MongoDB"""
    if current_date is None:
        current_date = datetime.now()
    
    print(f"\n{'='*80}")
    print(f"Analyzing data for {current_date.strftime('%Y-%m')}...")
    print(f"{'='*80}\n")
    
    # Engineer features for predictions
    df_engineered = engineer_features(df)
    
    # Get current and previous month data
    active_current, exits_current, hires_current = get_month_data(df, current_date)
    prev_month = current_date - relativedelta(months=1)
    active_previous, exits_previous, hires_previous = get_month_data(df, prev_month)
    
    # Calculate metrics
    total_employees = len(active_current)
    total_employees_prev = len(active_previous)
    exits_count = len(exits_current)
    exits_count_prev = len(exits_previous)
    hires_count = len(hires_current)
    hires_count_prev = len(hires_previous)
    
    # Attrition rate (monthly)
    attrition_rate = (exits_count / total_employees_prev * 100) if total_employees_prev > 0 else 0
    attrition_rate_prev = (exits_count_prev / len(active_previous) * 100) if len(active_previous) > 0 else 0
    
    # Make predictions on all data
    X = df_engineered.drop(columns=['EmployeeID', 'FullName', 'Attrition', 'TerminationMonthYear'], 
                           errors='ignore')
    probabilities = model.predict_proba(X)
    
    # Get department metrics
    dept_metrics = get_attrition_by_department(df, current_date)
    
    # Get top risk employees by department
    top_risk_by_dept = get_top_risk_employees_by_department(df, probabilities)
    
    # Get attrition timeline with predictions (current year only)
    attrition_timeline = predict_future_attrition(df, current_date)
    
    # Prepare document for MongoDB
    document = {
        'reportDate': current_date.strftime('%Y-%m'),
        'reportYear': current_date.year,
        'timestamp': datetime.utcnow(),
        'overallMetrics': {
            'totalEmployees': total_employees,
            'totalEmployeesChange': round(calculate_percentage_change(total_employees, total_employees_prev), 2),
            'exitsThisMonth': exits_count,
            'exitsChange': round(calculate_percentage_change(exits_count, exits_count_prev), 2),
            'newHiresThisMonth': hires_count,
            'newHiresChange': round(calculate_percentage_change(hires_count, hires_count_prev), 2),
            'attritionRateThisMonth': round(attrition_rate, 2),
            'attritionRateChange': round(calculate_percentage_change(attrition_rate, attrition_rate_prev), 2)
        },
        'departmentMetrics': dept_metrics,
        'topRiskEmployeesByDepartment': top_risk_by_dept,
        'attritionTimeline': attrition_timeline
    }
    
    # Store in MongoDB (replace if exists for same month)
    collection.replace_one(
        {'reportDate': current_date.strftime('%Y-%m')},
        document,
        upsert=True
    )
    
    print(f"{'='*80}")
    print("DATA STORED SUCCESSFULLY IN MONGODB")
    print(f"{'='*80}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Report Date: {current_date.strftime('%Y-%m')}")
    print(f"Report Year: {current_date.year}")
    print(f"\nOverall Metrics:")
    print(f"  Total Employees: {total_employees} ({document['overallMetrics']['totalEmployeesChange']:+.2f}%)")
    print(f"  Exits This Month: {exits_count} ({document['overallMetrics']['exitsChange']:+.2f}%)")
    print(f"  New Hires: {hires_count} ({document['overallMetrics']['newHiresChange']:+.2f}%)")
    print(f"  Attrition Rate: {attrition_rate:.2f}% ({document['overallMetrics']['attritionRateChange']:+.2f}%)")
    print(f"\nDepartments Analyzed: {len(dept_metrics)}")
    print(f"Top Risk Employees Identified: {sum(len(d['topRiskEmployees']) for d in top_risk_by_dept)}")
    print(f"Timeline Data Points: {len(attrition_timeline)} (Year {current_date.year} only)")
    
    # Show timeline breakdown
    actual_count = sum(1 for t in attrition_timeline if t['type'] == 'actual')
    predicted_count = sum(1 for t in attrition_timeline if t['type'] == 'predicted')
    print(f"  - Actual months: {actual_count}")
    print(f"  - Predicted months: {predicted_count}")
    
    return document


# ============================================================================
# EXECUTE ANALYSIS
# ============================================================================

if __name__ == "__main__":
    try:
        # Set the analysis date (change as needed)
        current_date = datetime(2025, 11, 1)  # November 2025

        # Run analysis and store
        document = analyze_and_store(df, model, current_date)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SAMPLE DATA SUMMARY")
        print(f"{'='*80}")
        
        print("\nDepartment Metrics (first 3):")
        for dept in document['departmentMetrics'][:3]:
            print(f"  {dept['department']}: {dept['totalEmployees']} employees, "
                  f"{dept['exitsYTD']} exits YTD, {dept['attritionRate']}% attrition")
        
        print("\nTop Risk Employees in First Department (first 3):")
        if document['topRiskEmployeesByDepartment']:
            first_dept = document['topRiskEmployeesByDepartment'][0]
            print(f"  Department: {first_dept['department']}")
            for emp in first_dept['topRiskEmployees'][:3]:
                print(f"    - {emp['fullName']} ({emp['jobRole']}): {emp['attritionProbability']}% risk")
        
        print(f"\nAttrition Timeline for {current_date.year}:")
        actual_months = [t for t in document['attritionTimeline'] if t['type'] == 'actual']
        predicted_months = [t for t in document['attritionTimeline'] if t['type'] == 'predicted']
        
        print(f"  Actual months: {len(actual_months)}")
        for entry in actual_months[:3]:
            print(f"    {entry['month']}: {entry['attritionRate']}% ({entry['attritionCount']} exits)")
        
        print(f"\n  Predicted months: {len(predicted_months)}")
        for entry in predicted_months:
            print(f"    {entry['month']}: {entry['attritionRate']}% "
                  f"[bounds: {entry['lowerBound']}-{entry['upperBound']}%]")
        
        print(f"\n{'='*80}")
        print("✓ Analysis complete and stored in MongoDB!")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR OCCURRED")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close MongoDB connection
        client.close()
        print("MongoDB connection closed.")