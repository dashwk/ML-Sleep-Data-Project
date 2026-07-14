# ML-Sleep-Data-Project
Personal Machine Learning Project: using Kaggle dataset of 100,000 rows of health/sleep data to predict if an individual sleeps well, comparing it to my own WHOOP data to test for accuracy

# Sleep Quality & Recovery Prediction

Predicting whether a person feels rested using lifestyle, behavioral, and physiological data, built as a supervised classification project, then applied to my own wearable (Whoop) data during a race training block as a personal case study.

## Motivation

While training for a 23-mile bike race, I wanted a way to quantify how lifestyle and recovery factors (sleep, stress, activity) related to feeling prepared and rested. Rather than train a model on my own limited personal data (not enough samples for honest validation), I trained a population-level classifier on a public dataset, then applied the trained model to my own Whoop data as a demo layer — comparing its predictions against Whoop's own proprietary recovery scoring.

## Dataset

[Sleep Health & Daily Performance Dataset](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset) — 100,000 records. Values are synthetic (privacy-preserving), but statistical relationships are calibrated against peer-reviewed sources including the CDC National Health Interview Survey (2024), Sleep Foundation population studies, *Frontiers in Sleep* (2025), and the Framingham Heart Study. 

**Target:** `felt_rested` (binary) — chosen over a manually-binarized sleep quality score because it's a natural, pre-labeled recovery/readiness outcome, closely aligned with what wearable devices like Whoop attempt to measure.

**Note on iteration:** an earlier version of this project used a smaller (400-record) sleep dataset. After deduplication, that dataset contained only ~109 unique records — too small for stable cross-validation (fold accuracy ranged 77%–100%). I sourced this larger dataset specifically to get a more statistically reliable result; the earlier version is preserved as `vi_sleep_model.ipynb` for reference.

## Features

12 features used, spanning behavioral and physiological variables:
`age`, `bmi`, `sleep_duration_hrs`, `caffeine_mg_before_bed`, `alcohol_units_before_bed`, `steps_that_day`, `nap_duration_mins`, `stress_score`, `work_hours_that_day`, `heart_rate_resting_bpm`, `weekend_sleep_diff_hrs`, `exercise_day`

Excluded as features due to leakage risk (too directly entangled with the target): `sleep_quality_score`, `rem_percentage`, `deep_sleep_percentage`, `cognitive_performance_score`, `sleep_disorder_risk`, `mental_health_condition`.

Also excluded: `wake_episodes_per_night`, `sleep_latency_mins`, `screen_time_before_bed_mins` — these had no reliable equivalent in my Whoop wearable data (see Personal Application section below), so they were dropped from both the training feature set and the model entirely, rather than approximated with weak proxies. An earlier version of this project used all 15 features; the 12-feature version is what's reported here for consistency between the trained model and the personal demo.

## Methodology

- Train/test split (80/20), stratified on target to preserve class balance (~61% / 39%)
- Two model families compared: **Logistic Regression** and **Random Forest**, each evaluated with default and `class_weight='balanced'` settings
- Hyperparameter tuning via `GridSearchCV` (5-fold cross-validation)
- Evaluation beyond accuracy: precision, recall, F1, confusion matrix — necessary given class imbalance, where accuracy alone can be misleading

## Results

| Model | Accuracy | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
|---|---|---|---|---|---|
| Logistic Regression (default) | 0.716 | 0.74 | 0.83 | 0.67 | 0.53 |
| Logistic Regression (balanced) | 0.716 | 0.80 | 0.72 | 0.62 | 0.71 |
| Random Forest (default) | 0.723 | 0.76 | 0.80 | 0.66 | 0.61 |
| Random Forest (balanced) | 0.718 | 0.79 | 0.73 | 0.62 | 0.69 |

All four models land within a percentage point of each other on accuracy (71.6%–72.3%), reinforcing that this ceiling reflects the limits of the feature set rather than any one model's weakness — a result that held even after trimming from 15 to 12 features. As before, the more meaningful lever was `class_weight='balanced'`, which traded a modest amount of class 0 recall for a substantial improvement in class 1 recall, rather than improving performance outright.

**Selected model: Logistic Regression (balanced).** In a wellness context, a false "you're rested" prediction is more consequential than the reverse — it risks pushing into a demanding day or workout without knowing the body needs recovery. Logistic regression (balanced) achieves the strongest class 1 recall (0.71) among all four variants while remaining the most interpretable model.

### Feature importance

Both models agree `sleep_duration_hrs` and `stress_score` are by far the two strongest predictors. They diverge on secondary features — Random Forest ranked `work_hours_that_day`, `steps_that_day`, `weekend_sleep_diff_hrs`, and `bmi` notably higher than Logistic Regression's coefficients suggested, consistent with these features having a nonlinear or threshold-based relationship with rest that a linear model underweights. This is a discussion point rather than a confirmed conclusion — worth further investigation (e.g., partial dependence plots) rather than an established fact.

## Personal application: Whoop data during race training

To make this project personal rather than purely academic, I mapped my own Whoop wearable data onto this feature set and ran it through the trained model across an 11-day window of training for a 23-mile bike race (August 2026), comparing its predictions against Whoop's own proprietary recovery scoring.

**Mapping wearable data to model features:** Whoop doesn't track identical fields to the training dataset, so several features required approximation, documented here rather than presented as like-for-like:
- `stress_score`: no direct Whoop equivalent; approximated as `100 - Recovery score %`, then min-max rescaled to match the training dataset's stress_score range, since the two used different underlying scales
- `caffeine_mg_before_bed`, `alcohol_units_before_bed`: Whoop's journal only records yes/no, not quantities — used as binary presence flags rather than true amounts
- `wake_episodes_per_night`, `sleep_latency_mins`, `screen_time_before_bed_mins`: no reliable wearable equivalent available; excluded from both the personal demo and the retrained model
- `steps_that_day`: pulled from phone step tracking for the same dates
- `work_hours_that_day`: logged manually against my actual schedule
- `weekend_sleep_diff_hrs`: computed directly from my own sleep data (weekday average minus weekend average) rather than treated as a per-day value
- One day had an unanswered Whoop journal entry; caffeine/alcohol were assumed absent (0) for that day

**Findings:** Comparing model predictions against Whoop's own recovery bands (green ≥67%, yellow/red <67%), the model agreed with Whoop's classification on 10 of 11 days (90.9%). The single disagreement (July 11, Whoop recovery 66%) fell directly on the boundary between Whoop's yellow and green bands — a borderline case rather than a clear miss.

The model's predicted probabilities also tracked Whoop's recovery score directionally: the three lowest-recovery days (44%, 45%, 51%) produced the model's three lowest and only sub-50%-probability predictions, and the highest-recovery day (94%) produced the highest predicted probability (0.93).

**Limitation:** the model showed less resolution in the "good recovery" middle range — Whoop scores from 66% to 82% all produced fairly similar, high predicted probabilities (0.75–0.93), suggesting the model captures the broad direction (good vs. poor recovery) well but doesn't resolve fine gradations the way Whoop's continuous, HRV-based score does. This comparison is illustrative and based on a small sample (11 days); the 90.9% figure reflects agreement with Whoop's own (also imperfect) proprietary score, not validated accuracy against ground truth, and shouldn't be read alongside the model's ~72% test-set accuracy as a directly comparable number.

## Limitations

- Dataset is synthetic; while relationships are calibrated to real research, this is not observed real-world data
- ~72% accuracy ceiling suggests these 12 features capture only part of what drives feeling rested — factors like illness, mental state, or environment aren't represented
- Personal demo section reflects a single individual over a short window and should be read as an illustrative case study, not a validated personalized prediction
- Class imbalance (61/39) was handled via `class_weight`, not resampling (e.g., SMOTE) — worth exploring as a future comparison

## Tech stack

Python, pandas, scikit-learn (LogisticRegression, RandomForestClassifier, GridSearchCV, Pipeline, StandardScaler), matplotlib/seaborn for visualization

## Future work

- Resampling techniques (SMOTE) as an alternative to class weighting
- Partial dependence plots to investigate nonlinear feature relationships flagged by Random Forest
- Expand personal demo dataset over a longer window post-race
- Test experimental features excluded from this pass (occupation, chronotype, season, shift work)
- Investigate model calibration in the mid-range of predicted probability, where resolution against Whoop's continuous score was weaker
