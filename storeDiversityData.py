import pandas as pd
import numpy as np
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
COLLECTION_NAME = "DiversityData"
CSV_FILE = "hr_synthetic.csv"

# Load data
df = pd.read_csv(CSV_FILE)

# MongoDB connection
client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Clear existing data
collection.delete_many({})
print("Cleared existing DiversityData collection")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_date(date_str):
    """Parse date string in YYYY-MM format"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        return datetime.strptime(str(date_str), '%Y-%m')
    except:
        return None


def categorize_age(age):
    """Categorize age into buckets"""
    if age < 26:
        return "18-25"
    elif age < 36:
        return "26-35"
    elif age < 46:
        return "36-45"
    elif age < 56:
        return "46-55"
    else:
        return "56+"


def categorize_tenure(years):
    """Categorize tenure into buckets"""
    if years < 1:
        return "<1"
    elif years < 3:
        return "1-3"
    elif years < 5:
        return "3-5"
    elif years < 10:
        return "5-10"
    else:
        return "10+"


# ============================================================================
# MAIN DIVERSITY ANALYSIS FUNCTION
# ============================================================================

def calculate_diversity_metrics(df, target_date):
    """Calculate diversity metrics for a specific month"""
    
    month_start = target_date.replace(day=1)
    next_month = month_start + relativedelta(months=1)
    
    # Parse dates
    df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
    df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)
    
    # Get active employees for this month
    active_employees = df[
        (df['JoiningDate'] < next_month) & 
        ((df['TerminationDate'].isna()) | (df['TerminationDate'] >= month_start))
    ].copy()
    
    if len(active_employees) == 0:
        print(f"No active employees for {target_date.strftime('%Y-%m')}")
        return None
    
    # Add categorization columns
    active_employees['AgeCategory'] = active_employees['Age'].apply(categorize_age)
    active_employees['TenureCategory'] = active_employees['YearsAtCompany'].apply(categorize_tenure)
    
    # 1. Gender Distribution
    gender_dist = active_employees['Gender'].value_counts().to_dict()
    
    # 2. Age Distribution
    age_dist = active_employees['AgeCategory'].value_counts().to_dict()
    # Sort age categories in the right order
    age_order = ["18-25", "26-35", "36-45", "46-55", "56+"]
    age_dist = {cat: age_dist.get(cat, 0) for cat in age_order}
    
    # 3. Tenure Distribution
    tenure_dist = active_employees['TenureCategory'].value_counts().to_dict()
    # Sort tenure categories in the right order
    tenure_order = ["<1", "1-3", "3-5", "5-10", "10+"]
    tenure_dist = {cat: tenure_dist.get(cat, 0) for cat in tenure_order}
    
    # 4. Diversity by Department (Gender breakdown per department)
    diversity_by_dept = {}
    for dept in active_employees['Department'].unique():
        dept_df = active_employees[active_employees['Department'] == dept]
        gender_counts = dept_df['Gender'].value_counts().to_dict()
        diversity_by_dept[dept] = gender_counts
    
    # Prepare document
    document = {
        'frequency': 'month',
        'period': target_date.strftime('%Y-%m'),
        'timestamp': datetime.utcnow(),
        'totalActiveEmployees': len(active_employees),
        'gender_distribution': gender_dist,
        'age_distribution': age_dist,
        'tenure_distribution': tenure_dist,
        'diversity_by_department': diversity_by_dept
    }
    
    return document


# ============================================================================
# GENERATE DIVERSITY DATA FOR MULTIPLE MONTHS
# ============================================================================

def generate_diversity_data(df, start_date, end_date=None):
    """Generate diversity data for a range of months"""
    
    if end_date is None:
        end_date = datetime.now()
    
    current_month = start_date.replace(day=1)
    documents_inserted = 0
    
    print(f"\n{'='*80}")
    print("GENERATING DIVERSITY DATA")
    print(f"{'='*80}\n")
    
    while current_month <= end_date:
        print(f"Processing {current_month.strftime('%Y-%m')}...", end=" ")
        
        document = calculate_diversity_metrics(df, current_month)
        
        if document:
            # Store in MongoDB (replace if exists for same month)
            collection.replace_one(
                {'period': current_month.strftime('%Y-%m')},
                document,
                upsert=True
            )
            documents_inserted += 1
            print(f"✓ Stored ({document['totalActiveEmployees']} employees)")
        else:
            print("✗ Skipped (no data)")
        
        current_month = current_month + relativedelta(months=1)
    
    return documents_inserted


# ============================================================================
# DISPLAY SUMMARY
# ============================================================================

def display_summary(document):
    """Display a summary of the diversity data"""
    
    print(f"\n{'='*80}")
    print(f"SAMPLE DIVERSITY DATA - {document['period']}")
    print(f"{'='*80}\n")
    
    print(f"Total Active Employees: {document['totalActiveEmployees']}")
    
    print("\n--- Gender Distribution ---")
    for gender, count in document['gender_distribution'].items():
        percentage = (count / document['totalActiveEmployees']) * 100
        print(f"  {gender}: {count} ({percentage:.1f}%)")
    
    print("\n--- Age Distribution ---")
    for age_group, count in document['age_distribution'].items():
        percentage = (count / document['totalActiveEmployees']) * 100
        print(f"  {age_group}: {count} ({percentage:.1f}%)")
    
    print("\n--- Tenure Distribution ---")
    for tenure_group, count in document['tenure_distribution'].items():
        percentage = (count / document['totalActiveEmployees']) * 100
        print(f"  {tenure_group} years: {count} ({percentage:.1f}%)")
    
    print("\n--- Diversity by Department (first 5) ---")
    for i, (dept, gender_counts) in enumerate(list(document['diversity_by_department'].items())[:5]):
        total_dept = sum(gender_counts.values())
        print(f"  {dept} (Total: {total_dept}):")
        for gender, count in gender_counts.items():
            percentage = (count / total_dept) * 100
            print(f"    {gender}: {count} ({percentage:.1f}%)")
    
    if len(document['diversity_by_department']) > 5:
        print(f"  ... and {len(document['diversity_by_department']) - 5} more departments")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        print(f"\n{'='*80}")
        print("DIVERSITY DATA GENERATION SCRIPT")
        print(f"{'='*80}\n")
        
        # Parse dates from the dataset to determine range
        df['JoiningDate'] = df['JoiningMonthYear'].apply(parse_date)
        df['TerminationDate'] = df['TerminationMonthYear'].apply(parse_date)
        
        # Get date range from data
        min_date = df['JoiningDate'].min()
        max_date = datetime.now()
        
        print(f"Data Range: {min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}")
        print(f"Total Employees in Dataset: {len(df)}")
        print(f"Active Employees: {len(df[df['Attrition'] == 'No'])}")
        print(f"Departed Employees: {len(df[df['Attrition'] == 'Yes'])}\n")
        
        # Generate diversity data for all months in range
        # You can customize the start date
        start_date = datetime(2020, 1, 1)  # Adjust as needed
        
        documents_count = generate_diversity_data(df, start_date, max_date)
        
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total documents inserted: {documents_count}")
        
        # Retrieve and display latest document
        latest_doc = collection.find_one(sort=[('timestamp', -1)])
        
        if latest_doc:
            display_summary(latest_doc)
            
            # Show sample document structure
            print(f"\n{'='*80}")
            print("SAMPLE MONGODB DOCUMENT STRUCTURE")
            print(f"{'='*80}\n")
            
            import json
            # Remove _id and timestamp for cleaner display
            display_doc = {k: v for k, v in latest_doc.items() if k not in ['_id', 'timestamp']}
            print(json.dumps(display_doc, indent=2, default=str))
        
        print(f"\n{'='*80}")
        print("✓ DIVERSITY DATA GENERATION COMPLETE")
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