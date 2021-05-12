# Predicting heart failure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

df= pd.read_csv('heart_failure_clinical_records_dataset-2.csv')
df.head()

import pandas_profiling as pp
pp.ProfileReport(df)

msno.matrix(df)
![image](https://user-images.githubusercontent.com/82114061/117928049-aa23ec00-b318-11eb-9245-b33f856eb8e5.png)

df.anaemia = df.anaemia.fillna('1')
df['creatinine_phosphokinase'].median()
df.creatinine_phosphokinase = df.creatinine_phosphokinase.fillna(250)
df.diabetes = df.diabetes.fillna('1')
df['ejection_fraction'].median()
df.ejection_fraction = df.ejection_fraction.fillna(38)
df['platelets'].median()
df.platelets = df.platelets.fillna(259500.0)
df['serum_sodium'].median()
df.serum_sodium = df.serum_sodium.fillna(137.0)
df.smoking = df.smoking.fillna('No')
df.head()


df['sex'].replace('Male',1)
df['sex'] = np.where(df['sex'].str.contains('Male'), 1, 0)
df['smoking'] = np.where(df['smoking'].str.contains('Yes I smoke'), 1, 0)
df['DEATH_EVENT'] = np.where(df['DEATH_EVENT'].str.contains("Death"), 1, 0)
df.head()

df1.loc[df1["sex"]== 0 , "sex"] = "female"
df1.loc[df1["sex"]== 1, "sex"] = "male"

df1.loc[df1["diabetes"]== 0 , "diabetes"] = "no diabetes"
df1.loc[df1["diabetes"]== 1, "diabetes"] = "diabetes"

df1.loc[df1['DEATH_EVENT'] == 0,'DEATH_EVENT'] = "LIVE"
df1.loc[df1['DEATH_EVENT'] == 1, 'DEATH_EVENT'] = 'DEATH'

fig = px.sunburst(df1, 
                  path=["sex","diabetes","DEATH_EVENT"],
                  values="count",
                  title="Gender & Diabetes Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
df1.loc[df1["smoking"]== 0 , "smoking"] = "non smoking"
df1.loc[df1["smoking"]== 1, "smoking"] = "smoking"

fig = px.sunburst(df1, 
                  path=["sex","smoking","DEATH_EVENT"],
                  values="count",
                  title="Gender & Smoking Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
df1.loc[df1["anaemia"]== 0 , "anaemia"] = "no anaemia"
df1.loc[df1["anaemia"]== 1, "anaemia"] = "anaemia"

fig = px.sunburst(df1, 
                  path=["sex","anaemia","DEATH_EVENT"],
                  values="count",
                  title="Gender & Anaemia  Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
df1.loc[df1["high_blood_pressure"]== 0 , "high_blood_pressure"] = "no high_blood_pressure"
df1.loc[df1["high_blood_pressure"]== 1, "high_blood_pressure"] = "high_blood_pressure"

fig = px.sunburst(df1, 
                  path=["sex","high_blood_pressure","DEATH_EVENT"],
                  values="count",
                  title="Gender & High Blood Pressure Sunburst Chart ",
                  width=600,
                  height=600)

fig.show()
![newplot (10)](https://user-images.githubusercontent.com/82114061/117928289-f66f2c00-b318-11eb-88c5-5a5fee378e49.png)
![newplot (11)](https://user-images.githubusercontent.com/82114061/117928326-06870b80-b319-11eb-9c3f-d295d48d1c17.png)
![newplot (12)](https://user-images.githubusercontent.com/82114061/117928340-0be45600-b319-11eb-9119-cb0639ddfaf1.png)
![newplot (13)](https://user-images.githubusercontent.com/82114061/117928354-13a3fa80-b319-11eb-8fda-2dad4745b9eb.png)


sex_mortality = []
sex_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['sex']==1)]))
sex_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['sex']==1)]))
sex_mortality.append(len(df[(df['DEATH_EVENT']==1)&(df['sex']==0)]))
sex_mortality.append(len(df[(df['DEATH_EVENT']==0)&(df['sex']==0)]))
sex_labels = ['male_died','male_survived','female_died','female_survived']

plt.pie(x=sex_mortality,autopct='%.1f',labels=sex_labels);
![image](https://user-images.githubusercontent.com/82114061/117928404-24547080-b319-11eb-8333-950d98c9d3c5.png)
![image](https://user-images.githubusercontent.com/82114061/117928426-2b7b7e80-b319-11eb-9976-b980202b343b.png)
![image](https://user-images.githubusercontent.com/82114061/117928439-30403280-b319-11eb-98df-3823da458904.png)
![image](https://user-images.githubusercontent.com/82114061/117928472-39c99a80-b319-11eb-9cdf-3ff525e9e4ce.png)
![image](https://user-images.githubusercontent.com/82114061/117928490-3fbf7b80-b319-11eb-92ac-c860be7b599d.png)


corr_matrix = df.corr()
sns.clustermap(corr_matrix, annot = True, fmt = ".2f")
plt.title("Correlaation btw features")
threshold = 0.2 
filtre = np.abs(corr_matrix["DEATH_EVENT"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(df[corr_features].corr(), annot = True, fmt = ".2f")
plt.title("Correlation Between Features w Corr Theshold 0.75")
plt.show()
![image](https://user-images.githubusercontent.com/82114061/117928566-5fef3a80-b319-11eb-8d6d-ec09530ca7a9.png)
![image](https://user-images.githubusercontent.com/82114061/117928579-64b3ee80-b319-11eb-9c58-e4f9b7c98062.png)


plt.figure(figsize=(15, 12))

plt.subplot(2,3,1)
sns.boxplot(x='DEATH_EVENT', y='age', data=df)
plt.title('Distribution of Age')

plt.subplot(2,3,2)
sns.boxplot(x='DEATH_EVENT', y='creatinine_phosphokinase', data=df)
plt.title('Distribution of creatinine_phosphokinase')

plt.subplot(2,3,3)
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=df)
plt.title('Distribution of ejection_fraction')

plt.subplot(2,3,4)
sns.boxplot(x='DEATH_EVENT', y='platelets', data=df)
plt.title('Distribution of platelets')

plt.subplot(2,3,5)
sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=df)
plt.title('Distribution of serum_creatinine')

plt.subplot(2,3,6)
sns.boxplot(x='DEATH_EVENT', y='serum_sodium', data=df)
plt.title('Distribution of serum_sodium');

![image](https://user-images.githubusercontent.com/82114061/117928767-a349a900-b319-11eb-8f42-8aeb2e7977b8.png)

df=df[df['creatinine_phosphokinase']<1300]
df=df[df['ejection_fraction']<60]
df=df[(df['platelets']>100000) & (df['platelets']<420000)]
df=df[df['serum_creatinine']<1.5]
df=df[df['serum_sodium']>126]

def displot_numeric_features(feature):#code to visualize distribution, scatterplot and boxplot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=110)
    
    sns.distplot(df[feature], ax=ax1)
    sns.scatterplot(df[feature], df["DEATH_EVENT"], ax=ax2)
    sns.boxplot(df[feature],orient='h', ax=ax3, width=0.2)

    print(f"Skewness Coefficient of {feature} is {df[feature].skew():.2f}")
    ax1.set_yticks([])
    
    return plt
    displot_numeric_features("creatinine_phosphokinase").show()
displot_numeric_features("ejection_fraction").show()
displot_numeric_features("platelets").show()
displot_numeric_features("serum_creatinine").show()
displot_numeric_features("serum_sodium").show()
![image](https://user-images.githubusercontent.com/82114061/117928967-e73cae00-b319-11eb-92e9-212450642df8.png)
![image](https://user-images.githubusercontent.com/82114061/117929038-f7ed2400-b319-11eb-82c9-53d441a9cf27.png)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('DEATH_EVENT',axis=1), 
                                                    df['DEATH_EVENT'], test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
confusion_matrix = pd.crosstab(y_test,predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
![image](https://user-images.githubusercontent.com/82114061/117929247-34b91b00-b31a-11eb-8959-d6940368ee87.png)

fig = px.histogram(df, x="time", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of TIME Vs DEATH_EVENT", 
                   labels={"time": "TIME"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()
fig = px.histogram(df, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of SERUM CREATININE Vs DEATH_EVENT", 
                   labels={"serum_creatinine": "SERUM CREATININE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()
fig = px.histogram(df, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of EJECTION FRACTION Vs DEATH_EVENT", 
                   labels={"ejection_fraction": "EJECTION FRACTION"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()
fig = px.histogram(df, x="age", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of AGE Vs DEATH_EVENT", 
                   labels={"age": "AGE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()
fig = px.histogram(df, x="creatinine_phosphokinase", color="DEATH_EVENT", marginal="violin", hover_data=df.columns, 
                   title ="Distribution of CREATININE PHOSPHOKINASE Vs DEATH_EVENT", 
                   labels={"creatinine_phosphokinase": "CREATININE PHOSPHOKINASE"},
                   template="plotly_dark",
                   color_discrete_map={"0": "RebeccaPurple", "1": "MediumPurple"}
                  )
fig.show()
![newplot (14)](https://user-images.githubusercontent.com/82114061/117929502-882b6900-b31a-11eb-9c24-2ddb28289a1f.png)
![newplot (15)](https://user-images.githubusercontent.com/82114061/117929516-8b265980-b31a-11eb-8391-087a45aeb167.png)
![newplot (16)](https://user-images.githubusercontent.com/82114061/117929536-8eb9e080-b31a-11eb-8113-5f73f0bb87ad.png)
![newplot (17)](https://user-images.githubusercontent.com/82114061/117929541-9083a400-b31a-11eb-9fb9-6c96b71ceb2d.png)
![newplot (18)](https://user-images.githubusercontent.com/82114061/117929549-937e9480-b31a-11eb-8611-ad132a3db11e.png)
