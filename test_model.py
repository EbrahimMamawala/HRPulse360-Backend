import pandas as pd
import joblib
import numpy as np

# Load the model dictionary
model_dict = joblib.load('best_pipeline.joblib')
model = model_dict['pipeline']

# Load the data
df = pd.read_csv('hr_synthetic.csv')

print("=" * 80)
print("MODEL TESTING SCRIPT - WITH FEATURE ENGINEERING")
print("=" * 80)

# Display model metadata
print("\n1. MODEL METADATA")
print("-" * 80)
print(f"Model Name: {model_dict['model_name']}")
print(f"CV Score: {model_dict['cv_score']:.4f}")
print(f"Best Parameters: {model_dict['best_params']}")
print("\nBaseline Scores (all models tested):")
for model_name, score in model_dict['baseline_scores'].items():
    print(f"  {model_name}: {score:.4f}")

# Data overview
print("\n2. DATA OVERVIEW")
print("-" * 80)
print(f"Dataset shape: {df.shape}")
print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")
print(f"Attrition rate: {df['Attrition'].value_counts()['Yes'] / len(df) * 100:.2f}%")

# Feature Engineering
print("\n3. FEATURE ENGINEERING")
print("-" * 80)

# Create engineered features
df['tenure_ratio'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
df['pay_per_experience'] = df['BasePay'] / (df['WorkExperienceYears'] + 1)
df['ManagerID_freq'] = df.groupby('ManagerID')['ManagerID'].transform('count')
df['overwork_index'] = df['AvgHoursWorkedPerMonth'] / 160

print("Created engineered features:")
print(f"  - tenure_ratio: YearsInCurrentRole / (YearsAtCompany + 1)")
print(f"  - pay_per_experience: BasePay / (WorkExperienceYears + 1)")
print(f"  - ManagerID_freq: Count of employees per manager")
print(f"  - overwork_index: AvgHoursWorkedPerMonth / 160")

# Prepare features
non_feature_cols = ['EmployeeID', 'FullName', 'Attrition', 'TerminationMonthYear']
X = df.drop(columns=[col for col in non_feature_cols if col in df.columns], errors='ignore')
y = df['Attrition']

# Convert y to numeric for consistency with model predictions
y_numeric = (y == 'Yes').astype(int)

print(f"\nTotal features after engineering: {X.shape[1]}")

print("\n4. MAKING PREDICTIONS")
print("-" * 80)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

print(f"Predictions made successfully!")
print(f"Prediction type: {type(predictions[0])}")
print(f"Unique prediction values: {np.unique(predictions)}")
print(f"Predicted 0 (No attrition): {(predictions == 0).sum()}")
print(f"Predicted 1 (Yes attrition): {(predictions == 1).sum()}")

# Performance metrics
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score

print("\n5. MODEL PERFORMANCE")
print("-" * 80)
print(f"Accuracy: {accuracy_score(y_numeric, predictions):.4f}")
print(f"Matthews Correlation Coefficient: {matthews_corrcoef(y_numeric, predictions):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_numeric, probabilities[:, 1]):.4f}")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_numeric, predictions)
print(f"                Predicted 0   Predicted 1")
print(f"                  (No)          (Yes)")
print(f"Actual 0 (No)   {cm[0][0]:>12}  {cm[0][1]:>13}")
print(f"Actual 1 (Yes)  {cm[1][0]:>12}  {cm[1][1]:>13}")

print(f"\nClassification Report:")
print(classification_report(y_numeric, predictions, target_names=['No', 'Yes']))

# Calculate additional metrics
true_negatives = cm[0][0]
false_positives = cm[0][1]
false_negatives = cm[1][0]
true_positives = cm[1][1]

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nAdditional Metrics:")
print(f"Precision (of 'Yes' predictions): {precision:.4f} - {true_positives}/{true_positives + false_positives} correct out of predicted attrition")
print(f"Recall/Sensitivity: {recall:.4f} - {true_positives}/{true_positives + false_negatives} actual attritions caught")
print(f"Specificity: {specificity:.4f} - {true_negatives}/{true_negatives + false_positives} correct non-attritions")
print(f"F1-Score: {f1:.4f}")

# High-risk employees
print("\n6. HIGH-RISK EMPLOYEES (Top 10)")
print("-" * 80)
attrition_proba = probabilities[:, 1]  # Probability of 1 (attrition)
high_risk_idx = np.argsort(attrition_proba)[-10:][::-1]

high_risk_df = df.iloc[high_risk_idx][['EmployeeID', 'FullName', 'Age', 'Department', 
                                         'YearsAtCompany', 'EngagementScore', 'Attrition']].copy()
high_risk_df['AttritionProb%'] = (attrition_proba[high_risk_idx] * 100).round(2)
high_risk_df['Prediction'] = ['Yes' if predictions[i] == 1 else 'No' for i in high_risk_idx]

print(high_risk_df.to_string(index=False))

# Low-risk employees who left (False Negatives - missed attritions)
print("\n7. MISSED ATTRITIONS (False Negatives with lowest probabilities)")
print("-" * 80)
false_negative_mask = (y_numeric == 1) & (predictions == 0)
if false_negative_mask.sum() > 0:
    fn_indices = np.where(false_negative_mask)[0]
    fn_probs = attrition_proba[fn_indices]
    lowest_fn_idx = fn_indices[np.argsort(fn_probs)[:5]]
    
    fn_df = df.iloc[lowest_fn_idx][['EmployeeID', 'FullName', 'Department', 
                                      'YearsAtCompany', 'EngagementScore']].copy()
    fn_df['AttritionProb%'] = (attrition_proba[lowest_fn_idx] * 100).round(2)
    fn_df['ActualLeft'] = 'Yes'
    fn_df['Predicted'] = 'No'
    print(fn_df.to_string(index=False))
    print(f"\nTotal False Negatives: {false_negative_mask.sum()} (missed {false_negative_mask.sum()/y_numeric.sum()*100:.2f}% of actual attritions)")
else:
    print("No false negatives!")

# Distribution of predictions
print("\n8. PREDICTION DISTRIBUTION")
print("-" * 80)
print(f"Average attrition probability: {attrition_proba.mean():.4f} ({attrition_proba.mean()*100:.2f}%)")
print(f"Median attrition probability: {np.median(attrition_proba):.4f} ({np.median(attrition_proba)*100:.2f}%)")
print(f"Max attrition probability: {attrition_proba.max():.4f} ({attrition_proba.max()*100:.2f}%)")
print(f"Min attrition probability: {attrition_proba.min():.4f} ({attrition_proba.min()*100:.2f}%)")

# Probability distribution
prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\nAttrition Probability Distribution:")
for i in range(len(prob_bins)-1):
    if i == len(prob_bins)-2:
        count = (attrition_proba >= prob_bins[i]).sum()
    else:
        count = ((attrition_proba >= prob_bins[i]) & (attrition_proba < prob_bins[i+1])).sum()
    print(f"  {prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}: {count:>6} employees ({count/len(df)*100:>5.2f}%)")

# Example prediction for a single employee
print("\n9. SINGLE EMPLOYEE PREDICTION EXAMPLE")
print("-" * 80)
sample_idx = 0
sample_employee = df.iloc[sample_idx]
sample_X = X.iloc[sample_idx:sample_idx+1]
sample_pred = model.predict(sample_X)[0]
sample_proba = model.predict_proba(sample_X)[0]

print(f"Employee ID: {sample_employee['EmployeeID']}")
print(f"Name: {sample_employee['FullName']}")
print(f"Age: {sample_employee['Age']}, Gender: {sample_employee['Gender']}")
print(f"Department: {sample_employee['Department']}, Role: {sample_employee['JobRole']}")
print(f"Years at Company: {sample_employee['YearsAtCompany']}, Years in Role: {sample_employee['YearsInCurrentRole']}")
print(f"Base Pay: ${sample_employee['BasePay']:.2f}")
print(f"Engagement Score: {sample_employee['EngagementScore']}, Performance: {sample_employee['PerformanceRating']}")
print(f"\nEngineered Features:")
print(f"  Tenure Ratio: {sample_employee['tenure_ratio']:.4f}")
print(f"  Pay per Experience: ${sample_employee['pay_per_experience']:.2f}")
print(f"  Overwork Index: {sample_employee['overwork_index']:.4f}")
print(f"\n{'='*40}")
print(f"Actual Attrition: {sample_employee['Attrition']}")
print(f"Predicted Attrition: {'Yes' if sample_pred == 1 else 'No'}")
print(f"Probability of Leaving: {sample_proba[1]:.2%}")
print(f"Probability of Staying: {sample_proba[0]:.2%}")
print(f"{'='*40}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)

# Save predictions to CSV
output_df = df[['EmployeeID', 'FullName', 'Department', 'JobRole', 'Attrition']].copy()
output_df['PredictedAttrition'] = ['Yes' if p == 1 else 'No' for p in predictions]
output_df['AttritionProbability'] = probabilities[:, 1]
output_df['RiskLevel'] = pd.cut(probabilities[:, 1], 
                                 bins=[0, 0.3, 0.6, 1.0], 
                                 labels=['Low', 'Medium', 'High'])
output_df['Correct'] = (output_df['Attrition'] == output_df['PredictedAttrition'])
output_df.to_csv('attrition_predictions.csv', index=False)
print(f"\nPredictions saved to 'attrition_predictions.csv'")
print(f"Columns: EmployeeID, FullName, Department, JobRole, Attrition, PredictedAttrition, AttritionProbability, RiskLevel, Correct")