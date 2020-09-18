import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import imblearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from boruta import BorutaPy
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

st.title('Data Mining Project')
st.title('Question3: Intelligent Decision-Making for Loan Application')
st.text('Chong Sheng Hua 1161104297')
st.text('Tee Choo Hwa 1161104595')
st.text('Ho Ting Fong 1161104246')


DATE_COLUMN = 'Unnamed: 0.1'
url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQucvrHOyvAx6ouyUcSpA4Z8H-vxMmybeAelXUR1_AcUBhdz2rMciA7UkFh-ELxiCoxmGHKP0eEYc0C/pub?gid=412485197&single=true&output=csv'


st.subheader('Load Bank_CS.csv')

data_load_state = st.text('Loading data...')
df = pd.read_csv(url, nrows=10000)
data_load_state.text("Done! (using st.cache)")


st.subheader('Raw data')
st.write(df)

st.header('Data Preprocessing')
data_load_state = st.text('Data Preprocessing...')
empty = df.isna().any()
empty = empty[empty.values == True].index.values

num_col_empty = [i for i in empty if df[i].dtype != 'object']
obj_col_empty = [i for i in empty if i not in num_col_empty]

st.subheader('Without filling N/A')
st.write(df[num_col_empty].describe())

st.subheader('Fill N/A with Mean')
# fill with mean
df1 = df.copy()

for i in num_col_empty:
    df1[i].fillna((df1[i].mean()), inplace=True) 

st.write(df1[num_col_empty].describe())

st.subheader('Fill N/A with Median')
# fill with median
df2 = df.copy()

for i in num_col_empty:
    df2[i].fillna((df2[i].median()), inplace=True)

st.write(df2[num_col_empty].describe())


st.subheader('Self Define Value')
st.write(' *Fill loan amount with the median of Loan_Tenure_Year group of each Employment_Type')
st.write(' *If both loan amount and loan tenure year is nan, then fill both with mean')
st.write(' *Others fill with mean (reason why mean chosen is because it give better std compared to median)')


# self define value
# fill loan amount with the median of Loan_Tenure_Year group of each Employment_Type
# if both loan amount and loan tenure year is nan, then fill both with mean
# others fill with mean (reason why mean chosen is because it give better std compared to median)


df3 = df.copy()

employment_type = df3['Employment_Type'].unique()
median_loan_by_year = pd.DataFrame(columns = ['Employment_Type', 'Loan_Tenure_Year', 'median_Loan_Amount'])

for i in employment_type:
    median = df3[df3['Employment_Type'] == i].groupby('Loan_Tenure_Year').Loan_Amount.median()
    temp = pd.DataFrame({'Employment_Type': i,'Loan_Tenure_Year':median.index, 'median_Loan_Amount':median.values})
    median_loan_by_year = pd.concat([median_loan_by_year, temp], ignore_index=True)
    
null_idx = df3[df3['Loan_Amount'].isna()].index.values

for i in null_idx:
    for j in range(len(median_loan_by_year)):
        if ((df3['Employment_Type'].iloc[i] == median_loan_by_year['Employment_Type'].iloc[j]) & (df3['Loan_Tenure_Year'].iloc[i] == median_loan_by_year['Loan_Tenure_Year'].iloc[j])):
            df3['Loan_Amount'].iloc[i] = median_loan_by_year['median_Loan_Amount'].iloc[j]

for i in num_col_empty:
    df3[i].fillna((df3[i].mean()), inplace=True)

st.write(df3[num_col_empty].describe())

st.write('*****')

st.write('Findings: since third method gives a slightly smaller std among the others, hence it is chosen for the fill na')

df = df3

column = [i for i in df1.columns if df1[i].dtype != 'object']

for i in column:
    df[i] = df[i].round(0).astype(int)

data_load_state.text("Done! 2.1: Fill N/A for numberical data")   


data_load_state = st.text('Data Preprocessing...')
# filter all string
df['State'] = df['State'].replace("\s", "", regex=True)
df['State'] = df['State'].replace("[^a-zA-Z]", "", regex=True)
df['State'] = df['State'].replace("JohorB", "Johor")
df['State'] = df['State'].replace(["NS", "NSembilan"], "Negeri Sembilan")
df['State'] = df['State'].replace(["KL", "KualaLumpur"], "Kuala Lumpur")
df['State'] = df['State'].replace("Trengganu", "Terengganu")
df['State'] = df['State'].replace("SWK", "Sarawak")
df['State'] = df['State'].replace(["Penang", "PulauPenang", "PPinang"], "Pulau Pinang")


# property type
# fill with mode by Employment_Type group of each State

property_by_location = pd.DataFrame(columns = ['State', 'Employment_Type', 'max_property'])
State = df.State.unique()
employment_type = df.Employment_Type.unique()

for i in State:
    for j in employment_type:
        max_property = df[((df['State'] == i) & (df['Employment_Type'] == j))].Property_Type.agg(lambda x:x.value_counts().index[0])
        if len(max_property) == 0:
            max_property = ''
        temp = pd.DataFrame({'State': [i],'Employment_Type':[j], 'max_property':[max_property]})
        property_by_location = pd.concat([property_by_location, temp], ignore_index=True)
    
new_df = df.copy()
null_idx = new_df[new_df['Property_Type'].isna()].index.values

for i in null_idx:
    for j in range(len(property_by_location)):
        if ((new_df['State'].iloc[i] == property_by_location['State'].iloc[j]) & (new_df['Employment_Type'].iloc[i] == property_by_location['Employment_Type'].iloc[j])):
            new_df['Property_Type'].iloc[i] = property_by_location['max_property'].iloc[j]


df = new_df

st.subheader('For Checking N/A')
st.write(df.isna().any())


# Credit_Card_types
# fill with mode by Employment_Type group of each State

cd_by_location = pd.DataFrame(columns = ['State', 'Employment_Type', 'max_type'])
State = df.State.unique()
employment_type = df.Employment_Type.unique()

for i in State:
    for j in employment_type:
        max_type = df[((df['State'] == i) & (df['Employment_Type'] == j))].Credit_Card_types.agg(lambda x:x.value_counts().index[0])
        if len(max_type) == 0:
            max_type = ''
        temp = pd.DataFrame({'State': [i],'Employment_Type':[j], 'max_type':[max_type]})
        cd_by_location = pd.concat([cd_by_location, temp], ignore_index=True)
    
new_df = df.copy()
null_idx = new_df[new_df['Credit_Card_types'].isna()].index.values

for i in null_idx:
    for j in range(len(cd_by_location)):
        if ((new_df['State'].iloc[i] == cd_by_location['State'].iloc[j]) & (new_df['Employment_Type'].iloc[i] == cd_by_location['Employment_Type'].iloc[j])):
            new_df['Credit_Card_types'].iloc[i] = cd_by_location['max_type'].iloc[j]


df = new_df

data_load_state.text('Done! 2.2: Fill N/A for categorical data')


data_load_state = st.text('Data Preprocessing...')
idx = df[df['Total_Income_for_Join_Application'] < df['Monthly_Salary']].index.values

for i in idx:
    df['Total_Income_for_Join_Application'].iloc[i] = df['Monthly_Salary'].iloc[i]

data_load_state.text('Done! 2.3: Replace unreasonable data')

#df[df['Total_Income_for_Join_Application'] < df['Monthly_Salary']]

# the data is not balance, so we need do perform data overdampling


st.subheader('The Data is not balance, so we need perform data oversampling')
st.text(df.Decision.value_counts())
here = df.Decision.value_counts()
st.bar_chart(here)

data_load_state = st.text('Data Preprocessing...')

df1 = df.copy()
num_col = [i for i in df1.columns if df1[i].dtype != 'object']
to_scale = df1[num_col]
min_max_scaler = MinMaxScaler()
scaled = min_max_scaler.fit_transform(to_scale)
df_scaled = pd.DataFrame(scaled, columns = num_col)
df1.drop(columns=num_col, inplace = True)
df1 = pd.concat([df1.loc[:,:'State'], df_scaled, df1.loc[:,'Decision':]], axis=1, sort=False)
#df1

data_load_state.text('Done! 2.4.1: Min-Max Normalization')

data_load_state = st.text('Data Preprocessing...')

# smote with one hot encoding

temp = pd.get_dummies(df1)
X = temp.drop(columns = ['Decision_Accept', 'Decision_Reject'], axis = 1)
y = temp.Decision_Accept

smt = imblearn.over_sampling.SMOTE(sampling_strategy="minority", random_state=42, k_neighbors=5)

X_res, y_res = smt.fit_resample(X, y)
df2 = pd.concat([X_res, y_res], axis=1, sort=False)
#df2 

data_load_state.text('Done! 2.4.2: Smote')

st.write('*******')

st.subheader('Class Distribution')

here = df2.Decision_Accept.value_counts()
here = here.rename(index={0: 'Reject', 1:'Accept'})
st.text(here)
st.bar_chart(here)

st.header('Exploratory Data Analysis')
st.subheader('Compare the statistics for numberical data before and after Smote')
st.write('*Before Smote')
st.write(df1[num_col].describe())
st.write('*After Smote')
st.write(df2[num_col].describe())

st.subheader('Exploratory Data Analysis before Smote')

st.text('1. Number of customer from each state and its Credit_Card_types count')

customer_location = df1.groupby(['State','Credit_Card_types']).Credit_Card_types.count().reset_index(name='Count')
gold = np.array(customer_location[customer_location['Credit_Card_types'] == 'gold'].Count)
normal = np.array(customer_location[customer_location['Credit_Card_types'] == 'normal'].Count)
platinum = np.array(customer_location[customer_location['Credit_Card_types'] == 'platinum'].Count)


gn = np.add(gold, normal).tolist()
 
r = [0,1,2,3,4,5,6,7,8]

plt.bar(r, gold, color='#7f6d5f', edgecolor='white', width=1, label = 'gold')
plt.bar(r, normal, bottom=gold, color='#557f2d', edgecolor='white', width=1, label = 'normal')
plt.bar(r, platinum, bottom=gn, color='#2d7f5e', edgecolor='white', width=1, label = 'platinum')
plt.legend()


plt.xticks(r, customer_location.State.unique(), fontweight='bold')
plt.title('Number of Customer In Each Location And Its Count For Credit Card Type')
plt.xlabel("State")
plt.xticks(rotation=90)
plt.ylabel('Count')
 
st.pyplot()

st.text('2. Number of customer from each state and their occupation')

customer_location = df1.groupby(['State','Employment_Type']).Employment_Type.count().reset_index(name='Count')
state = customer_location.State.unique()
employement = customer_location.Employment_Type.unique()

cl_df = pd.DataFrame(columns = ['State', 'Employment_Type'])
for i in range(len(state)):
    for j in range(len(employement)):
        cl_df.loc[len(cl_df)] = (state[i], employement[j])
cl_df = cl_df.merge(customer_location, on = ['State', 'Employment_Type'], how = 'outer')
cl_df.fillna(0, inplace = True)


Self_Employed = np.array(cl_df[cl_df['Employment_Type'] == 'Self_Employed'].Count)
employee = np.array(cl_df[cl_df['Employment_Type'] == 'employee'].Count)
employer = np.array(cl_df[cl_df['Employment_Type'] == 'employer'].Count)
government = np.array(cl_df[cl_df['Employment_Type'] == 'government'].Count)
Fresh_Graduate = np.array(cl_df[cl_df['Employment_Type'] == 'Fresh_Graduate'].Count)


r = [0,1,2,3,4,5,6,7,8]
bottom = 0

plt.bar(r, Self_Employed, color='red', edgecolor='white', width=1, label = 'Self_Employed')
bottom = np.add(bottom,Self_Employed).tolist()
plt.bar(r, employee, bottom=bottom, color='blue', edgecolor='white', width=1, label = 'employee')
bottom = np.add(bottom,employee).tolist()
plt.bar(r, employer, bottom=bottom, color='green', edgecolor='white', width=1, label = 'employer')
bottom = np.add(bottom,employer).tolist()
plt.bar(r, government, bottom=bottom, color='orange', edgecolor='white', width=1, label = 'government')
bottom = np.add(bottom,government).tolist()
plt.bar(r, Fresh_Graduate, bottom=bottom, color='purple', edgecolor='white', width=1, label = 'Fresh_Graduate')
plt.legend()


plt.xticks(r, cl_df.State.unique(), fontweight='bold')
plt.xlabel("State")
plt.xticks(rotation=90)
plt.ylabel('Count')
 
st.pyplot()

st.text('3. Loan amount statistic for each employment type')

# finding: goverment have a highest median loan amount might because of they have a more stable income source

temp = df1.copy()
column = num_col
for i in column:
    temp[i] =(temp[i] * (df[i].max() - df[i].min())) + df[i].min()
    temp[i] = temp[i].round(0).astype(int)

st.write(temp[temp['Decision'] == 'Accept'].groupby('Employment_Type').Loan_Amount.describe().sort_values('50%'))

fig, ax = plt.subplots(figsize=(10, 5))
g = sns.boxplot(data = temp[temp['Decision'] == 'Accept'], x = 'Employment_Type', y = 'Loan_Amount', ax = ax)
st.pyplot()

st.text('4. Is the income of customer is affortable for the loan installment for all accepted loan?')

temp['monthly_leftover'] = temp['Total_Income_for_Join_Application'] - (temp['Total_Sum_of_Loan'] / (temp['Years_to_Financial_Freedom']*12))
temp['affortability'] = temp['monthly_leftover'] - (temp['Loan_Amount'] / (temp['Loan_Tenure_Year']*12))

st.write('Ans: Percentage of customer that loan is accepted and has enough total income for the total loan installment (active loan + new loan):', (len(temp[((temp['Decision'] == 'Accept') & (temp['affortability']>=0))]) / len(temp[temp['Decision'] == 'Accept'])) *100)

st.text('5. Number of accept decision')

st.write(df1.Decision.value_counts())

st.subheader('Exploratory Data Analysis after Smote')

data_load_state = st.text('Convert back one hot variable into 1 column...')

url_1 = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSeR-FuUfKxE2hxeA3UkPQuwIbXr6BdryoH-a88EXtkb7sD67gIm1t6ay0IRbTgUawCfZ5t0IigPVR6/pub?gid=1092755098&single=true&output=csv'
df_smote = pd.read_csv(url_1, nrows=10000)

data_load_state.text('Done! (using st.cache)')

st.write(df_smote)

st.text('1. Number of customer from each state and its Credit_Card_types count')

customer_location = df_smote.groupby(['State','Credit_Card_types']).Credit_Card_types.count().reset_index(name='Count')
gold = np.array(customer_location[customer_location['Credit_Card_types'] == 'gold'].Count)
normal = np.array(customer_location[customer_location['Credit_Card_types'] == 'normal'].Count)
platinum = np.array(customer_location[customer_location['Credit_Card_types'] == 'platinum'].Count)


gn = np.add(gold, normal).tolist()
 
r = [0,1,2,3,4,5,6,7,8]

plt.bar(r, gold, color='#7f6d5f', edgecolor='white', width=1, label = 'gold')
plt.bar(r, normal, bottom=gold, color='#557f2d', edgecolor='white', width=1, label = 'normal')
plt.bar(r, platinum, bottom=gn, color='#2d7f5e', edgecolor='white', width=1, label = 'platinum')
plt.legend()


plt.xticks(r, customer_location.State.unique(), fontweight='bold')
plt.title('Number of Customer In Each Location And Its Count For Credit Card Type')
plt.xlabel("State")
plt.xticks(rotation=90)
plt.ylabel('Count')
 
st.pyplot()

st.text('2. Number of customer from each state and their occupation')

customer_location = df_smote.groupby(['State','Employment_Type']).Employment_Type.count().reset_index(name='Count')
state = customer_location.State.unique()
employement = customer_location.Employment_Type.unique()

cl_df = pd.DataFrame(columns = ['State', 'Employment_Type'])
for i in range(len(state)):
    for j in range(len(employement)):
        cl_df.loc[len(cl_df)] = (state[i], employement[j])
cl_df = cl_df.merge(customer_location, on = ['State', 'Employment_Type'], how = 'outer')
cl_df.fillna(0, inplace = True)


Self_Employed = np.array(cl_df[cl_df['Employment_Type'] == 'Self_Employed'].Count)
employee = np.array(cl_df[cl_df['Employment_Type'] == 'employee'].Count)
employer = np.array(cl_df[cl_df['Employment_Type'] == 'employer'].Count)
government = np.array(cl_df[cl_df['Employment_Type'] == 'government'].Count)
Fresh_Graduate = np.array(cl_df[cl_df['Employment_Type'] == 'Fresh_Graduate'].Count)


r = [0,1,2,3,4,5,6,7,8]
bottom = 0

plt.bar(r, Self_Employed, color='red', edgecolor='white', width=1, label = 'Self_Employed')
bottom = np.add(bottom,Self_Employed).tolist()
plt.bar(r, employee, bottom=bottom, color='blue', edgecolor='white', width=1, label = 'employee')
bottom = np.add(bottom,employee).tolist()
plt.bar(r, employer, bottom=bottom, color='green', edgecolor='white', width=1, label = 'employer')
bottom = np.add(bottom,employer).tolist()
plt.bar(r, government, bottom=bottom, color='orange', edgecolor='white', width=1, label = 'government')
bottom = np.add(bottom,government).tolist()
plt.bar(r, Fresh_Graduate, bottom=bottom, color='purple', edgecolor='white', width=1, label = 'Fresh_Graduate')
plt.legend()


plt.xticks(r, cl_df.State.unique(), fontweight='bold')
plt.xlabel("State")
plt.xticks(rotation=90)
plt.ylabel('Count')
 
st.pyplot()

st.text('3. Loan amount statistic for each employment type')

# finding: goverment have a highest median loan amount might because of they have a more stable income source

temp = df1.copy()
column = num_col
for i in column:
    temp[i] =(temp[i] * (df[i].max() - df[i].min())) + df[i].min()
    temp[i] = temp[i].round(0).astype(int)

st.write(temp[temp['Decision'] == 'Accept'].groupby('Employment_Type').Loan_Amount.describe().sort_values('50%'))

fig, ax = plt.subplots(figsize=(10, 5))
g = sns.boxplot(data = temp[temp['Decision'] == 'Accept'], x = 'Employment_Type', y = 'Loan_Amount', ax = ax)
st.pyplot()

st.text('4. Is the income of customer is affortable for the loan installment for all accepted loan?')

temp = df_smote.loc[:, ['Loan_Amount', 'Loan_Tenure_Year', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan',
                   'Total_Income_for_Join_Application', 'Decision_Accept']]

column = temp.loc[:,:'Total_Income_for_Join_Application'].columns

for i in column:
    temp[i] =(temp[i] * (df[i].max() - df[i].min())) + df[i].min()

temp['monthly_leftover'] = temp['Total_Income_for_Join_Application'] - (temp['Total_Sum_of_Loan'] / (temp['Years_to_Financial_Freedom']*12))
temp['affortability'] = temp['monthly_leftover'] - (temp['Loan_Amount'] / (temp['Loan_Tenure_Year']*12))

st.write('Ans: Percentage of customer that loan is accepted and \nhas enough total income for the total loan installment (active loan + new loan):', (len(temp[((temp['Decision_Accept'] == 1) & (temp['affortability']>=0))]) / len(temp[temp['Decision_Accept'] == 1])) *100)

st.text('5. number of accept decision')

st.write(df_smote['Decision_Accept'].value_counts())


st.subheader('Association Rule Mining')


st.subheader('Model Training')
st.text(' *predict accept or reject loan')

st.subheader('Features selection')
st.text(' *compare between BORUTA and RFE features selection')

y = df2.Decision_Accept
X = df2.drop("Decision_Accept", 1)
colnames = X.columns

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

rf = RandomForestClassifier(n_jobs = 1, class_weight = 'balanced', max_depth = 5)
rf.fit(X,y)
rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
rfe.fit(X,y)
rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

st.write(rfe_score)
st.subheader('RFE Features')
sns_rfe_plot = sns.catplot(x="Score", y="Features", data = rfe_score, kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("RFE Features")
st.pyplot()


rf = RandomForestClassifier(n_jobs = 1, class_weight = 'balanced', max_depth = 5)

feat_selector = BorutaPy(rf, n_estimators='auto', random_state = 1)
feat_selector.fit(X.values, y.values.ravel())
boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order = 1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns = ['Features', 'Score'])
boruta_score = boruta_score.sort_values('Score', ascending = False)
st.write(boruta_score)
st.subheader('Boruta Features')
sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:30], kind = "bar", 
               height=14, aspect=1.9, palette='coolwarm')
plt.title("Boruta Features")
st.pyplot()

st.write('In this case, rfe features selection is more suitable for us as it gives a more ralavant result for the features selection. in boruta, the features Loan_Amount, Score, Monthly_Salary has a score of 0, which is not making sense. while in rfe, these freatures has a score more than 0.70. hence, rfe features selection technique is chosen in our case.')

st.write("The features that is selected are:'Number_of_Side_Income','Score', 'Number_of_Loan_to_Approve'")


st.header('Classification')
st.write('We will be comparing among svm, DecisionTreeClassifier, random forest, Logistic Regression, Naïve Bayes, K-Nearest Neighbours, Kmean')

performance = pd.DataFrame(columns = ['model','df','Precision', 'Recall', 'F1', 'Accuracy'])


features1 = ['Number_of_Side_Income','Score', 'Number_of_Loan_to_Approve']

features2 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve','Loan_Tenure_Year'] 

features3 = ['Score', 'Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Number_of_Bank_Products', 'Loan_Tenure_Year', 
            'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan',
            'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Property_Type_bungalow', 
            'Property_Type_condominium', 'Property_Type_flat', 'Property_Type_terrace']

features4 = ['Score', 'Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Number_of_Bank_Products', 'Loan_Tenure_Year', 
            'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan',
            'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Years_for_Property_to_Completion', 
            'Number_of_Credit_Card_Facility', 'Property_Type_bungalow', 'Property_Type_condominium', 'Property_Type_flat', 
            'Property_Type_terrace']

features5 = ['Score', 'Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Number_of_Bank_Products', 'Loan_Tenure_Year', 
            'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan',
            'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents']

features6 = ['Number_of_Side_Income', 'Score', 'Number_of_Loan_to_Approve', 'Loan_Tenure_Year', 'Number_of_Bank_Products',
             'Total_Income_for_Join_Application', 'Monthly_Salary', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 'Loan_Amount',
             'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Number_of_Properties']

features7 = ['Score', 'Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Number_of_Bank_Products', 'Loan_Tenure_Year', 
            'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan',
            'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Property_Type_bungalow', 
            'Property_Type_condominium', 'Property_Type_flat', 'Property_Type_terrace', 'Years_for_Property_to_Completion']

features8 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 
              'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 
              'Loan_Amount', 'Credit_Card_Exceed_Months']

features9 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 
              'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 
              'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Property_Type_bungalow', 
            'Property_Type_condominium', 'Property_Type_flat', 'Property_Type_terrace', 'Years_for_Property_to_Completion']

features10 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 
              'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 
              'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Property_Type_bungalow', 
            'Property_Type_condominium', 'Property_Type_flat', 'Property_Type_terrace', 'Years_for_Property_to_Completion',
            'Employment_Type_Self_Employed', 'Employment_Type_government', 'Employment_Type_employer', 'Employment_Type_Fresh_Graduate']

features11 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 
              'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 
              'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents','Employment_Type_Self_Employed', 
             'Employment_Type_government', 'Employment_Type_employer', 'Employment_Type_Fresh_Graduate']

features12 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 
              'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 
              'Loan_Amount', 'Credit_Card_Exceed_Months', 'Employment_Type_Self_Employed', 'Employment_Type_government', 
             'Employment_Type_employer', 'Employment_Type_Fresh_Graduate']

features13 = ['Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 
              'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 
              'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Property_Type_bungalow', 
            'Property_Type_condominium', 'Property_Type_flat', 'Property_Type_terrace', 'Years_for_Property_to_Completion', 
             'Employment_Type_Self_Employed', 'Employment_Type_government', 
             'Employment_Type_employer', 'Employment_Type_Fresh_Graduate']

features14 = rfe_score[rfe_score['Score']> 0].Features.to_list()


# Split dataset into training set and test set
# 70-30 split and set random_state=1
d = 9
X = df2[features9]
y = df2.Decision_Accept

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# SVM

model_svm = svm.SVC(kernel='linear', gamma='auto', random_state = 10, probability=True)
model_svm.fit(X_train, y_train)
y_pred_SVM = model_svm.predict(X_test)

performance.loc[len(performance)] = ['svm', d, precision_score(y_test, y_pred_SVM).round(2), recall_score(y_test, y_pred_SVM).round(2), 
                                      f1_score(y_test, y_pred_SVM).round(2), accuracy_score(y_test, y_pred_SVM).round(2)]


# DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state = 10)

clf = clf.fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

performance.loc[len(performance)] = ['clf', d, precision_score(y_test, y_pred_clf).round(2), recall_score(y_test, y_pred_clf).round(2), 
                                     f1_score(y_test, y_pred_clf).round(2), accuracy_score(y_test, y_pred_clf).round(2)]


# random forest

rfc = RandomForestClassifier(random_state = 10)

rfc = rfc.fit(X_train, y_train)
y_pred_RFC = rfc.predict(X_test)

confusion_majority=confusion_matrix(y_test, y_pred_RFC)

performance.loc[len(performance)] = ['rfc', d, precision_score(y_test, y_pred_RFC).round(2), recall_score(y_test, y_pred_RFC).round(2), 
                                        f1_score(y_test, y_pred_RFC).round(2), accuracy_score(y_test, y_pred_RFC).round(2)]


# Logistic Regression

logreg = LogisticRegression(random_state = 10)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

confusion_majority=confusion_matrix(y_test, y_pred_lr)

performance.loc[len(performance)] = ['lr', d, precision_score(y_test, y_pred_lr).round(2), recall_score(y_test, y_pred_lr).round(2), 
                                     f1_score(y_test, y_pred_lr).round(2), accuracy_score(y_test, y_pred_lr).round(2)]


# Naive Bayes Classifier

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_NB = nb.predict(X_test)

confusion_majority=confusion_matrix(y_test, y_pred_NB)

performance.loc[len(performance)] = ['nb', d, precision_score(y_test, y_pred_NB).round(2), recall_score(y_test, y_pred_NB).round(2), 
                                     f1_score(y_test, y_pred_NB).round(2), accuracy_score(y_test, y_pred_NB).round(2)]


# K-Nearest Neighbours

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

confusion_majority=confusion_matrix(y_test, y_pred_knn)

performance.loc[len(performance)] = ['knn', d, precision_score(y_test, y_pred_knn).round(2), recall_score(y_test, y_pred_knn).round(2), 
                                     f1_score(y_test, y_pred_knn).round(2), accuracy_score(y_test, y_pred_knn).round(2)]

st.write(performance.sort_values(['Accuracy','F1','Precision','Recall'], ascending = False))


st.subheader('Performance')
X = df2[features9]
y = df2.Decision_Accept
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)



model_svm = svm.SVC(kernel='linear', gamma='auto', random_state = 10, probability=True).fit(X_train, y_train)
y_pred_SVM = model_svm.predict(X_test)

clf = DecisionTreeClassifier(random_state = 10).fit(X_train, y_train)
y_pred_clf = clf.predict(X_test)

rfc = RandomForestClassifier(random_state = 10).fit(X_train, y_train)
y_pred_RFC = clf.predict(X_test)

logreg = LogisticRegression(random_state = 10).fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

nb = GaussianNB().fit(X_train, y_train)
y_pred_NB = nb.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)




prob_svm = model_svm.predict_proba(X_test)
prob_svm = prob_svm[:,1]

prob_clf = clf.predict_proba(X_test)
prob_clf = prob_clf[:,1]

prob_rfc = rfc.predict_proba(X_test)
prob_rfc = prob_rfc[:,1]

prob_lr = logreg.predict_proba(X_test)
prob_lr = prob_lr[:,1]

prob_nb = nb.predict_proba(X_test)
prob_nb = prob_nb[:,1]

prob_knn = knn.predict_proba(X_test)
prob_knn = prob_knn[:,1]


fpr_svm, tpr_svm, threshold_svm = roc_curve(y_test, prob_svm)
fpr_clf, tpr_clf, threshold_clf = roc_curve(y_test, prob_clf)
fpr_rfc, tpr_rfc, threshold_rfc = roc_curve(y_test, prob_rfc)
fpr_lr, tpr_lr, threshold_lr = roc_curve(y_test, prob_lr)
fpr_nb, tpr_nb, threshold_nb = roc_curve(y_test, prob_nb)
fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, prob_knn)

plt.plot(fpr_svm, tpr_svm, color='orange', label='svm') 
plt.plot(fpr_clf, tpr_clf, color='blue', label='clf')  
plt.plot(fpr_rfc, tpr_rfc, color='purple', label='rfc')  
plt.plot(fpr_lr, tpr_lr, color='green', label='lr')  
plt.plot(fpr_nb, tpr_nb, color='yellow', label='nb')  
plt.plot(fpr_knn, tpr_knn, color='red', label='knn')  


plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()


st.pyplot()


st.header('Clustering')
st.subheader('Elbow Method')
st.write('In cluster analysis, the elbow method is a heuristic used in determining the number of clusters in a data set. The method consists of plotting the explained variation as a function of the number of clusters, and picking the elbow of the curve as the number of clusters to use.')
from sklearn.cluster import KMeans
cs = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    cs.append(kmeans.inertia_)
plt.plot(range(1, 11), cs)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('CS')
st.pyplot()

st.write('By the plot above, we can see that there is a kink at 3. Hence k = 2 can be considered a good number of the cluster to cluster this data.')

performance2 = pd.DataFrame(columns = ['method', 'df', 'Silhouette', 'Accuracy'])

d = 14
X = df2[features14]
y = df2.Decision_Accept


#K-mean clustering
kmeans = KMeans(n_clusters=2, random_state=10)
kmeans.fit(X)
#kmeans.cluster_centers_
#kmeans.inertia_
labels_km = kmeans.labels_

#Agglomerative clustering
agg = AgglomerativeClustering(n_clusters=2) 
agg.fit(X)
labels_agg = agg.labels_

#Birch clustering
bi = Birch(threshold = 0.01, n_clusters=2) 
bi.fit(X)
labels_bi = bi.labels_


correct_labels_km = sum(y == labels_km)
correct_labels_agg = sum(y == labels_agg)
correct_labels_bi = sum(y == labels_bi)

performance2.loc[len(performance2)] = [ 'kmean', d, round(silhouette_score(X, labels_km),2), round((correct_labels_km/y.size),2)]
performance2.loc[len(performance2)] = [ 'Agglomerative', d, round(silhouette_score(X, labels_agg),2), round((correct_labels_agg/y.size),2)]
performance2.loc[len(performance2)] = [ 'Birch', d, round(silhouette_score(X, labels_bi),2), round((correct_labels_bi/y.size),2)]

st.write(performance2.sort_values('Accuracy', ascending = False))

st.subheader('Conclusion')
st.write("RandomForestClassifier will be chosen as it gives the highest f1 score and accuracy among the other techniques. while for the features, features9, which are 'Number_of_Side_Income', 'Number_of_Loan_to_Approve', 'Score', 'Loan_Tenure_Year', 'Number_of_Bank_Products', 'Monthly_Salary', 'Total_Income_for_Join_Application', 'Years_to_Financial_Freedom', 'Total_Sum_of_Loan', 'Loan_Amount', 'Credit_Card_Exceed_Months', 'Number_of_Dependents', 'Property_Type_bungalow', 'Property_Type_condominium', 'Property_Type_flat', 'Property_Type_terrace', 'Years_for_Property_to_Completion' are selected for the model as these features give the highest recall, f1 score and accuracy.")

st.write('*******')

st.write(' *Accuracy on training set: 1.000 Accuracy on test set: 0.821 Mjority classifier Confusion Matrix')

st.write(' *Mjority TN= 376 Mjority FP= 166 Mjority FN= 24 Mjority TP= 496')

st.write('Precision= 0.75 Recall= 0.95 F1= 0.84 Accuracy= 0.82')

st.write('KMeans, Birch and Agglomerative clusterings get the 51% | accuracy')

st.write('*******')

st.write('Higher fn in bank loan can avoid approve loan to customer that possible couldnt pay the loan')

st.text('')