from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression



app = Flask(__name__)


def DecisionTreemodel():
    df = pd.read_csv('f2.csv')
    df = df.drop(['Unnamed: 0', 'native_country'], axis=1)
    df = df[df.occupation != ' ?']
    # select all categorical variables
    df_categorical = df.select_dtypes(include=['object']).columns

    # encode categorical variables using Label Encoder

    # select all categorical variables
    df_categorical = df.select_dtypes(include=['object'])

    # apply Label encoder to df_categorical

    le = preprocessing.LabelEncoder()
    df_categorical = df_categorical.apply(le.fit_transform)
    df_categorical.head()

    # concat df_categorical with original df
    df = df.drop(df_categorical.columns, axis=1)
    df = pd.concat([df, df_categorical], axis=1)

    df = df.drop(['capital_loss', 'capital_gain'], axis=1)

    # convert target variable income to categorical
    df['income'] = df['income'].astype('category')

    # Putting feature variable to X
    X = df.drop(['income', 'income_encoded', 'Sum', 'final_weight', 'education_num', 'occupation'], axis=1)
    print(X.columns)
    # Putting response variable to y
    y = df['income']
    dt_default = DecisionTreeClassifier(max_depth=5)
    dt_default.fit(X, y)

    return dt_default


def LinearRegressionModel():
    df = pd.read_csv('f2.csv')
    df = df.drop(['Unnamed: 0', 'native_country'], axis=1)
    df = df[df.occupation != ' ?']
    df_categorical = df.select_dtypes(include=['object'])

    # apply Label encoder to df_categorical

    le = preprocessing.LabelEncoder()
    df_categorical = df_categorical.apply(le.fit_transform)
    df_categorical.head()

    # concat df_categorical with original df
    df = df.drop(df_categorical.columns, axis=1)
    df = pd.concat([df, df_categorical], axis=1)

    y_train = df.pop('Sum')
    X_train = df
    X_train = X_train.drop(['education_num'], axis=1)

    # Running RFE with the output number of the variable equal to 10
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    rfe = RFE(lm, 5)  # running RFE
    rfe = rfe.fit(X_train, y_train)

    print(list(zip(X_train.columns, rfe.support_, rfe.ranking_)))

    return rfe


@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template("index.html")


@app.route("/first", methods=['GET', 'POST'])
def First():
    if request.method == 'POST':
        surfix = request.form.get('surfix')
        name = request.form.get('name')
        last_name = request.form.get('last_name')
        gender = request.form.get('gender')
        age = request.form.get('age')
        workclass = request.form.get('workclass')
        education = request.form.get('education')
        status = request.form.get('status')
        relationship = request.form.get('relationship')
        race = request.form.get('race')
        hours = request.form.get('hours')

        predictInputDecisionTree = pd.DataFrame([[education, status, relationship, race, age, hours, workclass, 0]],
                                    columns=['education', 'marital_status',
                                             'relationship', 'race', 'age', 'hours', 'workclass', 'sex'])

        print(predictInputDecisionTree)

        dt_model = DecisionTreemodel()
        y_pred_default = dt_model.predict(predictInputDecisionTree)

        print(y_pred_default)

        predictInputLinearRegression = pd.DataFrame([[education, status, relationship, race, 0]],
                                                columns=['education', 'marital_status',
                                                         'relationship', 'race', 'sex'])

        print(predictInputLinearRegression)

        #mlr_model = LinearRegressionModel()
        #y_pred_mlr = mlr_model.predict(predictInputLinearRegression)

        #print(y_pred_mlr)

        return render_template("second page.html",valueDt = y_pred_default)


app.run(debug=True)
