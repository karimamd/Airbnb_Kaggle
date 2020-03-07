import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from ndcg_scorer import ndcg_scorer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
classifiers = {'XGboost':0, 'LogisticRegression':1, 'DecisionTreeClassifier': 2,'MultinomialNB': 3, 'SVM':4, 'KNeighborsClassifier':5}
np.random.seed(0)

def load_n_preprocess():
    #Loading data
    df_train = pd.read_csv('train_users.csv')
    df_test = pd.read_csv('test_users.csv')
    labels = df_train['country_destination'].values
    df_train = df_train.drop(['country_destination'], axis=1)
    id_test = df_test['id']
    piv_train = df_train.shape[0]

    #Creating a DataFrame with train+test data
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    #Removing id and date_first_booking
    df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
    #Filling nan
    df_all = df_all.fillna(-1)

    #####Feature engineering#######
    #date_account_created
    dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
    df_all['dac_year'] = dac[:,0]
    df_all['dac_month'] = dac[:,1]
    df_all['dac_day'] = dac[:,2]
    df_all = df_all.drop(['date_account_created'], axis=1)

    #timestamp_first_active
    tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
    df_all['tfa_year'] = tfa[:,0]
    df_all['tfa_month'] = tfa[:,1]
    df_all['tfa_day'] = tfa[:,2]
    df_all = df_all.drop(['timestamp_first_active'], axis=1)

    #Age
    av = df_all.age.values
    df_all['age'] = np.where(np.logical_or(av<14, av>70), -1, av)

    #One-hot-encoding features
    ohe_feats = ['gender', 'signup_method', 'signup_flow',
     'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
      'signup_app', 'first_device_type', 'first_browser']
    #df_all.gender,unique() #get unique values of column
    for f in ohe_feats:
        df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
        df_all_dummy.head()
        df_all = df_all.drop([f], axis=1)
        df_all = pd.concat((df_all, df_all_dummy), axis=1)

    #Splitting train and test
    vals = df_all.values
    X = vals[:piv_train]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_test = vals[piv_train:]
    print('finished loading and preprocessing')
    return X, y, X_test, id_test, le

def train_n_submit(X, y, X_test, id_test, le, clf_id,grid):
    #Classifier
    clf = XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=30,
                         objective='multi:softprob', subsample=0.7, colsample_bytree=0.6, seed=0)

    clf_name='XGboost'
    if(clf_id == 1):
        clf= LogisticRegression(random_state=0,multi_class='auto')
        clf_name='LogisticRegression'
        # parameter tuning
        if(grid):
            # Create regularization penalty space
            penalty = ['l1', 'l2']
            # Create regularization hyperparameter space
            C = np.logspace(0, 4, 10)
            # Create hyperparameter options
            hyperparameters = dict(C=C, penalty=penalty)# Create regularization penalty space
            # Create grid search using 5-fold cross validation
            clf = GridSearchCV(clf, hyperparameters, cv=5,scoring=ndcg_scorer ,verbose=2)

    elif(clf_id == 2):
        clf= DecisionTreeClassifier()
        clf_name='DecisionTreeClassifier'

    elif(clf_id == 3):
        clf= MultinomialNB() #fill nan values by 0 instead of -1
        clf_name='MultinomialNB'

    elif(clf_id == 4):
        clf= SVC()
        clf_name='SVM'

    elif(clf_id == 5):
        clf= KNeighborsClassifier()
        clf_name='KNeighborsClassifier'


    print('Training starts: '+ clf_name)
    best_model=clf.fit(X, y)
    print('Training Ended')
    y_pred = best_model.predict_proba(X_test)
    if(grid):
        print("accuracy :",best_model.best_score_ * 100)
    else:
        print("Training accuracy : " + str(best_model.score(X,y))) #TODO change accuracy to real ev metric
    #Taking the 5 classes with highest probabilities
    ids = []  #list of ids
    cts = []  #list of countries
    for i in range(len(id_test)):
        idx = id_test[i]
        ids += [idx] * 5
        cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

    #Generate submission
    sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
    file_name = clf_name+'_sub.csv'
    sub.to_csv(file_name, index=False)
    print('output printed to file: '+file_name)
    return sub


if __name__ == "__main__":
    grid_search = False
    classifiers_with_gridsearch_code=[1] # add others to list after code setup
    # print classifier options
    for key , value in classifiers.items():
        print(key+" : "+str(value))
    clf_choice= int(input('choose the classifier number : '))
    if (clf_choice in classifiers_with_gridsearch_code):
        grid_choice = input('do you want to GridSearch hyperparams (y/n):')
        if(grid_choice == 'y'):
            grid_search=True
            print('Using Gridsearch with classifier')
        else:
            print('no GridSearch due to "n" or invalid input')
    #load data and make preprocessing
    X, y, X_test, id_test, le = load_n_preprocess()
    # Train model and submit results to file
    train_n_submit(X, y, X_test, id_test, le, clf_choice,grid=grid_search)
