from collections import Counter, defaultdict
from typing import Dict
from sklearn.impute import SimpleImputer
import pandas as pd

training_features = pd.read_csv("data/training_set_features.csv")

median_cols = [
    'h1n1_concern',
    'h1n1_knowledge',
    'behavioral_antiviral_meds',
    'behavioral_avoidance',
    'behavioral_face_mask',
    'behavioral_wash_hands',
    'behavioral_large_gatherings',
    'behavioral_outside_home',
    'behavioral_touch_face',
    'doctor_recc_h1n1',
    'doctor_recc_seasonal',
    'chronic_med_condition',
    'child_under_6_months',
    'health_worker',
    'health_insurance',
    'opinion_h1n1_vacc_effective',
    'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc',
    'opinion_seas_vacc_effective',
    'opinion_seas_risk',
    'opinion_seas_sick_from_vacc',
    'household_adults',
    'household_children'
]

most_frequent_cols = [
    'age_group',
    'education',
    'race',
    'sex',
    'income_poverty',
    'marital_status',
    'rent_or_own',
    'employment_status',
    'hhs_geo_region',
    'census_msa',
    'employment_industry',
    'employment_occupation'
]

median_imputer = SimpleImputer(strategy="median")
most_frequent_imputer = SimpleImputer(strategy="most_frequent")

training_features[median_cols] = median_imputer.fit_transform(training_features[median_cols])
training_features[most_frequent_cols] = most_frequent_imputer.fit_transform(training_features[most_frequent_cols])
print(training_features.isna().sum())

