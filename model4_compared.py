import math
import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
import time
import sys

class DualLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", buffering=1) 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = sys.stderr = DualLogger("model_training.log")

# -----------------------
# Config / quick hyperparams
# -----------------------
SEED = 42
np.random.seed(SEED)

# BSA tuning budget - adjust for quick debugging vs final runs
BSA_POP = 20         # population (birds)
BSA_ITERS = 50        # iterations (small for quick runs; increase for production)
BSA_N_JOBS = -1       # parallel workers for candidate eval (set >1 only if objective is single-threaded)
VERBOSE = True

# final top-K features to display
TOP_K = 30

# -----------------------
# Load dataset (from your generated CSV)
# -----------------------
DATA_PATH = "hr_synthetic.csv"
df = pd.read_csv(DATA_PATH)

drop_cols = ["JoiningMonthYear", "TerminationMonthYear"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

# -----------------------
# Feature selection & simple engineering (same as you had)
# -----------------------
features = [
    "Age","Gender","MaritalStatus","BasePay","Department","TeamSize","ContractType",
    "DistanceFromHome_km","HighestEducationLevel","EducationField","JobRole","NoOfCompaniesWorked",
    "WorkExperienceYears","OvertimeDone","LeavesTakenLastYear","AvgHoursWorkedPerMonth",
    "PercentSalaryHike","BonusPercent","EngagementScore","PerformanceRating","CountCoursesDoneLastYear",
    "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrentManager","ManagerID"
]
df = df[features + ["Attrition"]].copy()

df["tenure_ratio"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"].replace(0, 1))
df["overwork_index"] = (df["AvgHoursWorkedPerMonth"] / 160) * (df["OvertimeDone"] == "Yes")
df["pay_per_experience"] = df["BasePay"] / (df["WorkExperienceYears"].replace(0, 1))

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

y = (df["Attrition"] == "Yes").astype(int)
X = df.drop(columns=["Attrition"])

# -----------------------
# Categorical encoding: OHE for low-cardinality, freq for high-cardinality
# -----------------------
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

HIGH_CARD_THRESHOLD = 20
high_card_cols = [c for c in cat_cols if X[c].nunique() > HIGH_CARD_THRESHOLD]
low_card_cols = [c for c in cat_cols if c not in high_card_cols]

for col in high_card_cols:
    freq = X[col].value_counts(normalize=True)
    X[col + "_freq"] = X[col].map(freq).fillna(0.0)
    X.drop(columns=[col], inplace=True)

num_cols = [c for c in X.columns if X[c].dtype.kind in 'bifc']
cat_cols = [c for c in X.columns if X[c].dtype == object]

# -----------------------
# Preprocessor (median imputer + StandardScaler for numeric; OHE for categorical)
# -----------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
    ("ohe", ohe),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
], remainder="drop")

# -----------------------
# Train/test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# -----------------------
# Bird Swarm Algorithm (BSA) implementation (Mantegna Levy-flight)
# -----------------------
class BirdSwarm:
    def __init__(self, n, function, lb, ub, dimension, iterations,
                 p_foraging=0.7, alpha=0.5, beta=0.3, flocking_weight=0.1,
                 levy_prob=0.05, vmax=None, n_jobs=1, seed=None, verbose=False):
#       Notes for reference:
#       α (alpha): weight for moving toward best solution.
#       β (beta): randomness intensity for vigilant birds.
        self.function = function
        self.n = int(n)
        self.dimension = int(dimension)
        self.iterations = int(iterations)
        self.p_foraging = float(p_foraging)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.flocking_weight = float(flocking_weight)
        self.levy_prob = float(levy_prob)
        self.n_jobs = int(n_jobs)
        self.verbose = bool(verbose)
        self.rng = np.random.RandomState(seed)

        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        if self.lb.shape == ():
            self.lb = np.full(self.dimension, float(lb))
        if self.ub.shape == ():
            self.ub = np.full(self.dimension, float(ub))

        # initialize population
        self._agents = self.rng.uniform(self.lb, self.ub, size=(self.n, self.dimension))

        if vmax is None:
            self.vmax = 0.2 * (self.ub - self.lb)
        else:
            self.vmax = np.array(vmax, dtype=float)
            if self.vmax.shape == ():
                self.vmax = np.full(self.dimension, float(vmax))

        # initial evaluation
        self.scores = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(self._safe_eval)(self._agents[i]) for i in range(self.n)
        ))
        best_idx = int(np.argmin(self.scores))
        self.gbest_pos = self._agents[best_idx].copy()
        self.gbest_score = float(self.scores[best_idx])
        self.history = {"gbest_score": [self.gbest_score]}

        for it in range(1, self.iterations + 1):
            centroid = np.mean(self._agents, axis=0)
            num_foragers = max(1, int(np.round(self.p_foraging * self.n)))
            idxs = np.arange(self.n)
            self.rng.shuffle(idxs) # shuffle to randomize forager/vigilant selection
            forager_idxs = idxs[:num_foragers]
            vigilant_idxs = idxs[num_foragers:]

            # foragers
            for i in forager_idxs:
                step = self.alpha * (self.gbest_pos - self._agents[i]) # alpha is step size factor
                step += self.rng.normal(scale=0.01, size=self.dimension) * (self.ub - self.lb) 
                flock_step = self.flocking_weight * (centroid - self._agents[i])
                move = step + flock_step
                if self.rng.rand() < self.levy_prob:
                    move += self._levy_flight(scale=0.01 * (self.ub - self.lb))
                move = np.clip(move, -self.vmax, self.vmax)
                self._agents[i] = np.clip(self._agents[i] + move, self.lb, self.ub)

            # vigilants
            for i in vigilant_idxs:
                rng_step = self.rng.normal(scale=1.0, size=self.dimension) * (self.ub - self.lb) * self.beta
                attract = 0.1 * (self.gbest_pos - self._agents[i])
                move = rng_step + attract
                if self.rng.rand() < self.levy_prob:
                    move += self._levy_flight(scale=0.05 * (self.ub - self.lb))
                move = np.clip(move, -self.vmax, self.vmax)
                self._agents[i] = np.clip(self._agents[i] + move, self.lb, self.ub)

            # evaluate
            scores = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(self._safe_eval)(self._agents[i]) for i in range(self.n)
            ))
            cur_best_idx = int(np.argmin(scores))
            cur_best_score = float(scores[cur_best_idx])
            if cur_best_score < self.gbest_score:
                self.gbest_score = cur_best_score
                self.gbest_pos = self._agents[cur_best_idx].copy()

            self.history["gbest_score"].append(self.gbest_score)
            if self.verbose and (it % max(1, self.iterations // 10) == 0 or it == 1):
                print(f"[BSA] iter={it}/{self.iterations} gbest_score={self.gbest_score:.6f}")

        if self.verbose:
            print("[BSA] finished. best_score=", self.gbest_score)

    def _safe_eval(self, x):
        try:
            return float(self.function(x))
        except Exception as e:
            if self.verbose:
                print("[BSA] objective eval error:", e)
            return np.inf

    def _levy_flight(self, scale=0.01):
        # Mantegna's algorithm
        beta = 1.5
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (num / den) ** (1 / beta)
        u = self.rng.normal(0, sigma_u, self.dimension)
        v = self.rng.normal(0, 1, self.dimension)
        step = u / (np.abs(v) ** (1 / beta))
        s = np.array(scale, dtype=float)
        if s.shape == ():
            s = np.full(self.dimension, float(s))
        return step * s

    def get_best(self):
        return self.gbest_pos

    def get_best_score(self):
        return self.gbest_score

    def get_history(self):
        return self.history

# -----------------------
# Objective functions (negated mean CV F1)
# CV uses n_jobs=1 to avoid nested parallelism
# -----------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

def objective_rf(params):
    n_estimators = int(round(float(params[0])))
    max_depth_raw = float(params[1])
    max_depth = int(round(max_depth_raw)) if max_depth_raw > 0 else None
    model = RandomForestClassifier(
        n_estimators=max(10, n_estimators),
        max_depth=max_depth,
        random_state=SEED,
        n_jobs=1,
        class_weight='balanced'
    )
    pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
    return -float(score)

def objective_lr(params):
    C = float(np.atleast_1d(params)[0])
    C = max(1e-6, C)
    model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=SEED, class_weight='balanced')
    pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
    return -float(score)

def objective_xgb(params):
    n_estimators = int(round(float(params[0])))
    max_depth = max(1, int(round(float(params[1]))))
    learning_rate = float(params[2])
    subsample = float(params[3])
    colsample = float(params[4])
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = max(1.0, float(neg) / float(max(1, pos)))
    model = XGBClassifier(
        n_estimators=max(10, n_estimators),
        max_depth=max_depth,
        learning_rate=max(1e-5, min(1.0, learning_rate)),
        subsample=max(0.3, min(1.0, subsample)),
        colsample_bytree=max(0.3, min(1.0, colsample)),
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
        scale_pos_weight=scale_pos_weight
    )
    pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
    return -float(score)

def objective_lgbm(params):
    # params: [n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree]
    n_estimators = int(round(float(params[0])))
    max_depth = int(round(float(params[1])))
    learning_rate = float(params[2])
    num_leaves = int(round(float(params[3])))
    subsample = float(params[4])
    colsample = float(params[5])
    model = LGBMClassifier(
        n_estimators=max(10, n_estimators),
        max_depth=max(-1, max_depth),
        learning_rate=max(1e-5, min(1.0, learning_rate)),
        num_leaves=max(8, min(512, num_leaves)),
        subsample=max(0.3, min(1.0, subsample)),
        colsample_bytree=max(0.3, min(1.0, colsample)),
        random_state=SEED,
        n_jobs=1,
        class_weight='balanced',
        boosting_type='gbdt',
        verbose=-1
    )
    pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
    return -float(score)

def objective_cat(params):
    # params: [iterations, depth, learning_rate, l2_leaf_reg]
    iterations = int(round(float(params[0])))
    depth = int(round(float(params[1])))
    lr = float(params[2])
    l2 = float(params[3])
    model = CatBoostClassifier(
        iterations=max(10, iterations),
        depth=max(1, depth),
        learning_rate=max(1e-5, min(1.0, lr)),
        l2_leaf_reg=max(0.0, l2),
        verbose=0,
        random_seed=SEED,
        thread_count=1,
        auto_class_weights='Balanced'
    )
    pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", model),
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
    return -float(score)

def objective_brf(params):
    # params: [n_estimators, max_depth]
    n_estimators = int(round(float(params[0])))
    max_depth = int(round(float(params[1])))
    model = BalancedRandomForestClassifier(
        n_estimators=max(10, n_estimators),
        max_depth=(max_depth if max_depth > 0 else None),
        random_state=SEED,
        n_jobs=1
    )
    # Note: BalancedRandomForest does its own under-sampling, so SMOTE is redundant.
    pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("model", model),
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
    return -float(score)

# -----------------------
# Search spaces (bounds) for each model
# -----------------------
search_space = {
    "XGBoost": (objective_xgb, [50, 2, 0.01, 0.5, 0.5], [300, 10, 0.3, 1.0, 1.0], 5),
    "RandomForest": (objective_rf, [50, 2], [300, 30], 2),
    "LogisticRegression": (objective_lr, [1e-6], [10.0], 1),
    "LightGBM": (objective_lgbm, [50, 3, 0.01, 16, 0.5, 0.5], [500, 12, 0.3, 256, 1.0, 1.0], 6),
    "CatBoost": (objective_cat, [50, 3, 0.01, 0.0], [500, 10, 0.3, 10.0], 4),
    "BalancedRF": (objective_brf, [50, 2], [300, 30], 2),
}
# -----------------------
# Calculate the performance of each model with its default parameters to establish a baseline.
# -----------------------
print("\n--- Calculating Baseline Scores (Default Hyperparameters) ---\n")
baseline_scores = {}
for model_name in search_space.keys():
    print(f"Calculating baseline for {model_name}...")
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=1)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(random_state=SEED, class_weight='balanced', max_iter=1000)
    elif model_name == "XGBoost":
        pos = np.sum(y_train == 1); neg = np.sum(y_train == 0)
        scale_pos_weight = max(1.0, float(neg) / float(max(1, pos)))
        model = XGBClassifier(random_state=SEED, scale_pos_weight=scale_pos_weight, n_jobs=1)
    elif model_name == "LightGBM":
        model = LGBMClassifier(random_state=SEED, class_weight='balanced', n_jobs=1, verbose=-1)
    elif model_name == "CatBoost":
        model = CatBoostClassifier(random_seed=SEED, auto_class_weights='Balanced', verbose=0, thread_count=1)
    elif model_name == "BalancedRF":
        model = BalancedRandomForestClassifier(random_state=SEED, n_jobs=1)

    # BalancedRF has its own sampling, so we don't use SMOTE with it for the baseline
    if model_name == "BalancedRF":
         pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])
    else:
        pipeline = ImbPipeline(steps=[
            ("pre", preprocessor),
            ("smote", SMOTE(random_state=SEED)),
            ("model", model),
        ])

    score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
    baseline_scores[model_name] = score
    print(f"--> {model_name} Baseline CV F1: {score:.4f}\n")

# -----------------------
# Run BSA for each model
# -----------------------
best_results = {}
for model_name, (func, lb, ub, dim) in search_space.items():
    print(f"\nRunning BSA for {model_name} ...")
    optimizer = BirdSwarm(
        n=BSA_POP,
        function=func,
        lb=lb,
        ub=ub,
        dimension=dim,
        iterations=BSA_ITERS,
        p_foraging=0.7,
        alpha=0.5,
        beta=0.35,
        flocking_weight=0.08,
        levy_prob=0.05,
        vmax=None,
        n_jobs=BSA_N_JOBS,
        seed=SEED,
        verbose=VERBOSE,
    )
    best_params = optimizer.get_best()
    best_score = -func(best_params)
    best_results[model_name] = (best_params, best_score)
    print(f"--> {model_name} Baseline F1: {baseline_scores[model_name]:.4f} | Optimized F1: {best_score:.4f} | Improvement: {best_score - baseline_scores[model_name]:.4f}")
    print(f"    Best Params: {np.round(best_params, 4)}")


# -----------------------
# Choose best model & retrain on full training set
# -----------------------
best_model_name = max(best_results, key=lambda k: best_results[k][1])
best_params, best_cv_score = best_results[best_model_name]
print("\n" + "="*50)
print(f"🏆 Overall Best Model: {best_model_name}")
print(f"   - Baseline CV F1:     {baseline_scores[best_model_name]:.4f}")
print(f"   - Optimized CV F1:    {best_cv_score:.4f}")
print(f"   - Raw Best Params: {best_params}")
print("="*50 + "\n")


# build final estimator from best params
if best_model_name == "RandomForest":
    final_model = RandomForestClassifier(
        n_estimators=int(round(float(best_params[0]))),
        max_depth=(int(round(float(best_params[1]))) if float(best_params[1]) > 0 else None),
        random_state=SEED,
        n_jobs=-1,
        class_weight='balanced'
    )
elif best_model_name == "LogisticRegression":
    final_model = LogisticRegression(
        C=float(np.atleast_1d(best_params)[0]),
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
        class_weight='balanced'
    )
elif best_model_name == "XGBoost":
    n_est = int(round(float(best_params[0])))
    depth = int(round(float(best_params[1])))
    lr = float(best_params[2])
    subs = float(best_params[3])
    colsm = float(best_params[4])
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = max(1.0, float(neg) / float(max(1, pos)))
    final_model = XGBClassifier(
        n_estimators=max(10, n_est),
        max_depth=max(1, depth),
        learning_rate=max(1e-5, min(1.0, lr)),
        subsample=max(0.3, min(1.0, subs)),
        colsample_bytree=max(0.3, min(1.0, colsm)),
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
elif best_model_name == "LightGBM":
    n_est = int(round(float(best_params[0])))
    depth = int(round(float(best_params[1])))
    lr = float(best_params[2])
    leaves = int(round(float(best_params[3])))
    subs = float(best_params[4])
    colsm = float(best_params[5])
    final_model = LGBMClassifier(
        n_estimators=max(10, n_est),
        max_depth=max(-1, depth),
        learning_rate=max(1e-5, min(1.0, lr)),
        num_leaves=max(8, leaves),
        subsample=max(0.3, min(1.0, subs)),
        colsample_bytree=max(0.3, min(1.0, colsm)),
        random_state=SEED,
        n_jobs=-1,
        class_weight='balanced',
        verbose=-1
    )
elif best_model_name == "CatBoost":
    iterations = int(round(float(best_params[0])))
    depth = int(round(float(best_params[1])))
    lr = float(best_params[2])
    l2 = float(best_params[3])
    final_model = CatBoostClassifier(
        iterations=max(10, iterations),
        depth=max(1, depth),
        learning_rate=max(1e-5, min(1.0, lr)),
        l2_leaf_reg=max(0.0, l2),
        verbose=0,
        random_seed=SEED,
        thread_count=-1,
        auto_class_weights='Balanced'
    )
else:  # BalancedRF
    n_est = int(round(float(best_params[0])))
    depth = int(round(float(best_params[1])))
    final_model = BalancedRandomForestClassifier(
        n_estimators=max(10, n_est),
        max_depth=(depth if depth > 0 else None),
        random_state=SEED,
        n_jobs=-1
    )

# Build final pipeline
if best_model_name == "BalancedRF":
    final_pipeline = Pipeline(steps=[("pre", preprocessor), ("model", final_model)])
else:
    final_pipeline = ImbPipeline(steps=[
        ("pre", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("model", final_model),
    ])

# Fit on training set
print("Training final model on full training set...")
final_pipeline.fit(X_train, y_train)

# Save pipeline + metadata for future use
save_path = "best_pipeline.joblib"
joblib.dump({
    "pipeline": final_pipeline,
    "model_name": best_model_name,
    "best_params": best_params,
    "cv_score": best_cv_score,
    "baseline_scores": baseline_scores
}, save_path)
print(f"Saved trained pipeline + metadata to '{save_path}'")

# baseline evaluation (default threshold 0.5)
y_pred = final_pipeline.predict(X_test)
print("\nFinal model performance at threshold=0.5:")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# If predict_proba is available, tune threshold to maximize F1 on test set
try: #review
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")
    best_t = 0.5
    best_f1 = f1_score(y_test, y_pred)
    for t in np.linspace(0.01, 0.99, 99):
        pred_t = (y_proba >= t).astype(int)
        f1t = f1_score(y_test, pred_t)
        if f1t > best_f1:
            best_f1 = f1t
            best_t = t
    print(f"Best threshold (by F1) on test set: {best_t:.2f} -> F1={best_f1:.4f}")
    pred_best = (y_proba >= best_t).astype(int)
    print("Precision (best t):", precision_score(y_test, pred_best))
    print("Recall (best t):", recall_score(y_test, pred_best))
except Exception as e:
    print("predict_proba not available or failed:", e)

# -----------------------
# Feature importance (if tree-based)
# -----------------------
model = final_pipeline.named_steps["model"]
if hasattr(model, "feature_importances_"):
    fitted_pre = final_pipeline.named_steps["pre"]
    feature_names = []
    # Get numeric feature names
    feature_names.extend(num_cols)
    # Get OHE feature names
    if "cat" in fitted_pre.named_transformers_ and fitted_pre.named_transformers_["cat"] != 'drop' and len(cat_cols) > 0:
        try:
            ohe_obj = fitted_pre.named_transformers_["cat"].named_steps["ohe"]
            cat_feature_names = ohe_obj.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(cat_feature_names)
        except Exception as e:
            print(f"Could not get categorical feature names: {e}")

    importances = model.feature_importances_
    if len(importances) == len(feature_names):
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        print("\nTop features by importance:")
        print(imp_df.head(TOP_K))
    else:
        print(f"\nFeature importance length ({len(importances)}) doesn't match feature names length ({len(feature_names)}).")
else:
    print("Final model does not expose feature_importances_ (e.g., logistic).")

test_acc = final_pipeline.score(X_test, y_test)
print("\nFinal Test Accuracy:", test_acc)
print("\nDone.")

log_file.close()