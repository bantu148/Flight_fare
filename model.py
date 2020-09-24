from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
sns.set()
train_data = pd.read_excel(r"Data_Train.xlsx")
pd.set_option('display.max_columns', None)
train_data.head()
train_data.isnull().sum()
train_data.dropna(inplace=True)


train_data['Journey_day'] = pd.to_datetime(
    train_data.Date_of_Journey, format="%d/%m/%Y").dt.day
train_data['Journey_month'] = pd.to_datetime(
    train_data.Date_of_Journey, format="%d/%m/%Y").dt.month
train_data.drop(['Date_of_Journey'], axis=1, inplace=True)
# extrating Hour
train_data['Dep_Hour'] = pd.to_datetime(train_data.Dep_Time).dt.hour

# extrating minute
train_data['Dep_Minute'] = pd.to_datetime(train_data.Dep_Time).dt.minute
train_data.drop(['Dep_Time'], axis=1, inplace=True)
# extrating Arrival Hour
train_data['Arrival_hour'] = pd.to_datetime(train_data.Arrival_Time).dt.hour

# extrating Arrival Minute
train_data['Arrival_minute'] = pd.to_datetime(
    train_data.Arrival_Time).dt.minute
train_data.drop(['Arrival_Time'], axis=1, inplace=True)
# converting Duration into list
duration = list(train_data['Duration'])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:  # check if duration containd hour or minute
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"  # Add 0 minute
        else:
            duration[i] = "0h " + duration[i]  # Add 0 hour


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep="h")[0]))
    duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins
Airline = train_data[['Airline']]
Airline = pd.get_dummies(Airline, drop_first=True)
Airline.head()
Source = train_data[['Source']]
Source = pd.get_dummies(Source, drop_first=True)
Source.head()
Destination = train_data[['Destination']]
Destination = pd.get_dummies(Destination, drop_first=True)
Destination.head()
train_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)
train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2,
                    "3 stops": 3, "4 stops": 4}, inplace=True)
data_train = pd.concat([train_data, Airline, Source, Destination], axis=1)
data_train.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
data_train.drop(['Duration'], axis=1,  inplace=True)
xa = data_train['Total_Stops']
xb = data_train.iloc[:, 2:]
x = pd.concat([xa, xb], axis=1)
y = data_train.iloc[:, 1]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=33)
reg_rf = RandomForestRegressor()
reg_rf.fit(x_train, y_train)
y_pred = reg_rf.predict(x_test)

# RandomizedSearch CV
# numbers of trees in random forest

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# number of features to consider at every split
max_features = ['auto', 'sqrt']

# maximum numbers of levels in trees
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

# minimun number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

# minimun number of sample required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid,
                               scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(x_train, y_train)
prediction = rf_random.predict(x_test)
filename = "joblibmodel.sav"
joblib.dump(reg_rf, filename)
