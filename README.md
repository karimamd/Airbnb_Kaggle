# Airbnb new user bookings kaggle challenge
https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/

# Problem Statment:
Using the dataset provided build a model that can predict a new user’s destination.

# Data
* The dataset is one with 15 user features and a label column for training data.

* The are 12 possible outcomes for a sample lable 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL', 'DE', 'AU', 'NDF' (no destination found), and 'other'. 


* The dataset features are : 
  'id', 'date_account_created', 'timestamp_first_active',   'date_first_booking', 'gender', 'age', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser', 'country_destination'

# Evaluation Metric:
The evaluation metric for this competition is NDCG (Normalized discounted cumulative gain) @k where k=5.

# Benchmark
* NDCG of 0.88697 is the best score on Kaggle
* Got NDCG of  0.85253 for dummy submission based on EDA !!!


# Exploratory Analysis
Light-weight exploration notebook available but all the plots are made in Metabase (https://www.metabase.com/start/)
after conveting dataset to sqlite
plots images are in slides: 
https://docs.google.com/presentation/d/1Q8MjFmean-9Ksv-b7uO42d08Gjtee5qNCtsqqar3I0g/edit?usp=sharing


# Instructions
* clone the repo
* make sure you have python 3.6+, pandas , numpy, sklearn, xgboost
* run : "python run.py" in terminal and follow instructions
