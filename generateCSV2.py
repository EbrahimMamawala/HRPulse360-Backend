import numpy as np
import pandas as pd

RND = 42
np.random.seed(RND)
N_ROWS = 50000

def clamp(arr, low, high):
    return np.minimum(np.maximum(arr, low), high)

# ---------- Basic identity fields ----------
employee_id = np.arange(1, N_ROWS + 1)
first_names = ["Alekya", "Samar", "Jayant", "Tarun", "Kris", "Mohan", "Kashyap", "Rahil", "Jaimin", "Avni",
               "Parikshit", "Kiran", "Lakshya", "Dhruv", "Bhavik", "Devansh", "Sana", "Priya", "Rahul"]
last_names = ["Shah", "Patel", "Khan", "Singh", "Gupta", "Iyer", "Nair", "Das", "Verma", "Bose",
              "Mehta", "Rao", "Kumar", "Kapoor", "Malhotra", "Nambiar", "Desai", "Mistry", "Joshi", "Chaudhary"]
full_name = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(N_ROWS)]

# ---------- Demographics ----------
age = clamp(np.round(np.random.normal(loc=33, scale=8, size=N_ROWS)).astype(int), 18, 60)
gender = np.random.choice(["Male", "Female", "Other"], size=N_ROWS, p=[0.6, 0.38, 0.02])
marital = np.random.choice(["Single", "Married", "Divorced", "Widowed"], size=N_ROWS, p=[0.45, 0.50, 0.04, 0.01])

# ---------- Org structure ----------
departments = ["Engineering", "HR", "Sales", "Finance", "Operations", "Product", "Support", "Marketing"]
job_roles = {
    "Engineering": ["Software Engineer", "SWE-Sr", "Data Engineer", "Platform Engineer"],
    "HR": ["HR Manager", "Talent Specialist"],
    "Sales": ["Sales Rep", "Sales Manager", "Account Executive"],
    "Finance": ["Accountant", "Finance Manager", "Controller"],
    "Operations": ["Ops Specialist", "Ops Manager"],
    "Product": ["Product Manager", "Product Designer"],
    "Support": ["Support Engineer", "Customer Success"],
    "Marketing": ["Marketing Specialist", "Growth Manager"]
}
dept = np.random.choice(departments, size=N_ROWS, p=[0.25, 0.06, 0.12, 0.08, 0.12, 0.10, 0.15, 0.12])
job_role = [np.random.choice(job_roles[d]) for d in dept]

contract = np.random.choice(["Permanent", "Contract"], size=N_ROWS, p=[0.85, 0.15])
team_size = np.maximum(1, (np.random.poisson(lam=6, size=N_ROWS) + np.array([5 if "Manager" in jr else 0 for jr in job_role])))

# ---------- Compensation ----------
base_pay = np.zeros(N_ROWS)
for i in range(N_ROWS):
    d = dept[i]
    jr = job_role[i]
    base = {
        "Engineering": 900,
        "Product": 850,
        "Sales": 700,
        "Finance": 700,
        "Operations": 600,
        "Support": 450,
        "HR": 500,
        "Marketing": 650
    }[d]
    if "Sr" in jr or "Manager" in jr or "Lead" in jr:
        base *= 1.4
    exp_factor = 1 + max((age[i] - 22) / 40, 0) * 0.6
    base_pay[i] = np.round(np.random.normal(loc=base * exp_factor, scale=base * 0.08), 2)

# ---------- Location / education / experience ----------
distance = clamp(np.round(np.random.exponential(scale=8, size=N_ROWS)), 0, 80).astype(int)
edu_levels = ["10th", "12th", "Bachelors", "Masters", "PhD"]
edu_probs = [0.01, 0.05, 0.65, 0.25, 0.04]
education = np.random.choice(edu_levels, size=N_ROWS, p=edu_probs)
education_field = np.random.choice(["CS/IT", "Business", "Engineering", "Arts", "Science", "Other"], size=N_ROWS, p=[0.35,0.18,0.20,0.12,0.1,0.05])

no_of_companies = np.maximum(0, np.random.poisson(lam=1.8, size=N_ROWS))
work_exp = clamp((age - 22) - np.random.poisson(1, size=N_ROWS), 0, 40).astype(int)
work_exp = np.maximum(work_exp, no_of_companies + 0)

# ---------- Work patterns ----------
overtime = np.random.choice(["Yes", "No"], size=N_ROWS, p=[0.22, 0.78])
leaves = np.maximum(0, np.random.poisson(lam=8 - (0.5 * (overtime == "Yes")), size=N_ROWS)).astype(int)
avg_hours = clamp(np.random.normal(loc=170 + (overtime == "Yes") * 20, scale=20, size=N_ROWS), 60, 320).round(1)

pct_hike = clamp(np.round(np.random.normal(loc=8, scale=4, size=N_ROWS), 2), -5, 40)
bonus_pct = clamp(np.round(np.random.normal(loc=8, scale=5, size=N_ROWS), 2), 0, 50)

engagement = clamp(np.round(np.random.normal(loc=3.4, scale=0.9, size=N_ROWS), 1), 1.0, 5.0)
perf_rating = clamp(np.round(np.random.normal(loc=3.2, scale=0.8, size=N_ROWS), 1), 1.0, 5.0)
courses_done = np.random.poisson(lam=1.2, size=N_ROWS)

# ---------- Joining dates: realistic distribution across 2020-2025 ----------
current_date = pd.Timestamp("2025-10-01")
all_month_starts = pd.date_range("2020-01-01", current_date, freq="MS")
year_weights = {2020: 0.05, 2021: 0.15, 2022: 0.25, 2023: 0.25, 2024: 0.20, 2025: 0.10}
month_probs = np.array([year_weights[d.year] for d in all_month_starts])
month_probs = month_probs / month_probs.sum()
# sample from pandas DatetimeIndex -> convert to pandas Timestamps explicitly
join_dates = pd.to_datetime(np.random.choice(all_month_starts, size=N_ROWS, p=month_probs))

# ---------- Years at company derived from joining date (bounded by work_exp) ----------
years_at_company = np.array([
    (current_date.year - jd.year) - (1 if current_date.month < jd.month else 0)
    for jd in join_dates
])
years_at_company = np.maximum(years_at_company, 0)
years_at_company = np.minimum(years_at_company, work_exp)

# ---------- Role/manager related time fields ----------
years_in_current_role = np.minimum(np.random.poisson(lam=np.maximum(1, years_at_company * 0.5), size=N_ROWS), years_at_company)
years_since_promo = np.minimum(np.random.poisson(lam=2, size=N_ROWS), np.maximum(0, years_in_current_role))
years_with_manager = np.minimum(np.random.poisson(lam=2, size=N_ROWS), years_at_company)

# ---------- Manager biases ----------
num_managers = 200
manager_ids = [f"M{str(i).zfill(4)}" for i in np.random.randint(1, num_managers + 1, size=N_ROWS)]
manager_bias_map = {f"M{str(i).zfill(4)}": np.random.normal(loc=0.0, scale=0.2) for i in range(1, num_managers + 1)}
manager_bias = np.array([manager_bias_map[m] for m in manager_ids])

# ---------- Attrition risk model ----------
coef = {
    "intercept": -1.2,
    "age": -0.01,
    "distance": 0.02,
    "base_pay_log": -0.6,
    "engagement": -1.0,
    "pct_hike": -0.03,
    "bonus_pct": -0.02,
    "leaves": 0.02,
    "no_of_companies": 0.12,
    "overtime_yes": 0.15,
    "contract_is": 0.4,
    "years_at_company": -0.05,
    "perf_rating": -0.12,
    "courses_done": -0.03,
    "years_since_promo": 0.05
}

base_pay_log = np.log1p(base_pay)
risk = coef["intercept"] \
       + coef["age"] * age \
       + coef["distance"] * distance \
       + coef["base_pay_log"] * base_pay_log \
       + coef["engagement"] * engagement \
       + coef["pct_hike"] * pct_hike \
       + coef["bonus_pct"] * bonus_pct \
       + coef["leaves"] * leaves \
       + coef["no_of_companies"] * no_of_companies \
       + coef["overtime_yes"] * (overtime == "Yes").astype(float) \
       + coef["contract_is"] * (contract == "Contract").astype(float) \
       + coef["years_at_company"] * years_at_company \
       + coef["perf_rating"] * perf_rating \
       + coef["courses_done"] * courses_done \
       + coef["years_since_promo"] * years_since_promo \
       + manager_bias

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

prob_attrition = sigmoid(risk)
# calibrate to ~15% attrition
current_mean = prob_attrition.mean()
target = 0.15
adjust = np.log(target / (1 - target)) - np.log(current_mean / (1 - current_mean))
prob_attrition = sigmoid(risk + adjust)
attrition = np.random.binomial(1, prob_attrition, size=N_ROWS)
attrition_label = np.where(attrition == 1, "Yes", "No")

# ---------- Termination dates (only for attrited employees) ----------
termination_dates = []
for i in range(N_ROWS):
    if attrition[i] == 1:
        jd = join_dates[i]  # pandas Timestamp
        # earliest realistic termination: 1 month after join
        months_between = (current_date.year - jd.year) * 12 + (current_date.month - jd.month)
        # ensure at least 1 month
        if months_between <= 0:
            term = current_date
        else:
            # pick a random month offset from join (1 .. months_between)
            offset_months = np.random.randint(1, months_between + 1)
            term = (jd + pd.DateOffset(months=int(offset_months))).replace(day=1)
            # don't exceed current_date start-of-month
            if term > current_date:
                term = current_date.replace(day=1)
        termination_dates.append(term)
    else:
        termination_dates.append(pd.NaT)

# ---------- Formatting month-year strings ----------
join_monthyear = [ts.strftime('%Y-%m') for ts in join_dates]
termination_monthyear = [ts.strftime('%Y-%m') if not pd.isna(ts) else '' for ts in termination_dates]

# ---------- Build DataFrame ----------
df = pd.DataFrame({
    "EmployeeID": employee_id,
    "FullName": full_name,
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital,
    "Attrition": attrition_label,
    "BasePay": base_pay.round(2),
    "Department": dept,
    "TeamSize": team_size,
    "ContractType": contract,
    "DistanceFromHome_km": distance,
    "HighestEducationLevel": education,
    "EducationField": education_field,
    "JobRole": job_role,
    "NoOfCompaniesWorked": no_of_companies,
    "WorkExperienceYears": work_exp,
    "OvertimeDone": overtime,
    "LeavesTakenLastYear": leaves,
    "AvgHoursWorkedPerMonth": avg_hours,
    "PercentSalaryHike": pct_hike,
    "BonusPercent": bonus_pct,
    "EngagementScore": engagement,
    "PerformanceRating": perf_rating,
    "CountCoursesDoneLastYear": courses_done,
    "YearsAtCompany": years_at_company,
    "YearsInCurrentRole": years_in_current_role,
    "YearsSinceLastPromotion": years_since_promo,
    "YearsWithCurrentManager": years_with_manager,
    "ManagerID": manager_ids,
    "JoiningMonthYear": join_monthyear,
    "TerminationMonthYear": termination_monthyear
})

# ---------- Save & quick checks ----------
out_path = "hr_synthetic.csv"
df.to_csv(out_path, index=False)

class_balance = df["Attrition"].value_counts(normalize=True).to_dict()
print("Saved:", out_path)
print("Class balance:", class_balance)
