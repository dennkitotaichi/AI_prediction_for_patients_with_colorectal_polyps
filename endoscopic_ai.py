import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import codecs
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Read data
image_file_path = './simulated_dpc_data.csv'
with codecs.open(image_file_path, "r", "Shift-JIS", "ignore") as file:
    dpc = pd.read_table(file, delimiter=",")

# dpc_r, g_dpc_r_1, g_r: restricted data from dpc
dpc_r=dpc.loc[:, ['ID','code']]
# g_dpc_r_1: made to check the details (: name of the code, ‘name’)
g_dpc_r_1=dpc.loc[:, ['ID','code','name']]
# Dummy Encoding with ‘name’
g_r = pd.get_dummies(dpc_r['code'])

# Reconstruct simulated data for AI learning
df_concat_dpc_get_dummies = pd.concat([dpc_r, g_r], axis=1)
# Remove features that may be the cause of the data leak
dpc_Remove_data_leak = df_concat_dpc_get_dummies.drop(["code",160094710,160094810,160094910,150285010,2113008,8842965,8843014,622224401,810000000,160060010], axis=1)
# Sum up the number of occurrences of each feature for each patient.
total_patient_features= dpc_Remove_data_leak.groupby("ID").sum()
total_patient_features.reset_index()
# Load a new file with ID and treatment availability

# Prepare training data
image_file_path_ID_and_polyp_pn = './simulated_patient_data.csv'
with codecs.open(image_file_path_ID_and_polyp_pn, "r", "Shift-JIS", "ignore") as file:
    ID_and_polyp_pn = pd.read_table(file, delimiter=",")
ID_and_polyp_pn_data= ID_and_polyp_pn[['ID', 'target']]
#Combine the new file containing ID and treatment status with the file after dummy encoding by the ‘name’
ID_treatment_medical_statement=pd.merge(ID_and_polyp_pn_data,total_patient_features,on=["ID"],how='outer')
ID_treatment_medical_statement_o= ID_treatment_medical_statement.fillna(0)
ID_treatment_medical_statement_p=ID_treatment_medical_statement_o.drop("ID", axis=1)
ID_treatment_medical_statement_rename= ID_treatment_medical_statement_p.rename(columns={'code':"Receipt type code"})
merge_data= ID_treatment_medical_statement_rename


# Split the training/validation set into 80% and the test set into 20%, with a constant proportion of cases with lesions
X = merge_data.drop("target",axis=1).values
y = merge_data["target"].values
columns_name = merge_data.drop("target",axis=1).columns
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=1)
# Create a function to divide data
def data_split(X,y):
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        X_train = pd.DataFrame(X_train, columns=columns_name)
        X_test = pd.DataFrame(X_test, columns=columns_name)
        return X_train, y_train, X_test, y_test
# Separate into training, validation, and test set
X_train, y_train, X_test, y_test = data_split(X, y)
X_train, y_train, X_val, y_val = data_split(X_train.values, y_train)
# Make test set into pandas
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
# Make test set into test_df to keep away for the final process
test_dfp = pd.concat([y_test_df,X_test_df], axis=1)
test_df=test_dfp.rename(columns={0:"target"})
# Make training/validation sets into pandas
y_trainp = pd.DataFrame(y_train)
X_trainp = pd.DataFrame(X_train)
train=pd.concat([y_trainp, X_trainp], axis=1)
y_valp = pd.DataFrame(y_val)
X_valp = pd.DataFrame(X_val)
val=pd.concat([y_valp, X_valp], axis=1)
test_vol=pd.concat([train, val])
training_validation_sets=test_vol.rename(columns={0:"target"})

# Create a function to save the results and feature importance after analysis with lightGBM
def reg_top10_lightGBM(merge_data,outname,no,random_state_number):
    # Define the objective variable
    X = merge_data.drop("target",axis=1).values
    y = merge_data["target"].values
    columns_name = merge_data.drop("target",axis=1).columns
    # Define a function
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state_number)
    def data_split(X,y):
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        X_train = pd.DataFrame(X_train, columns=columns_name)
        X_test = pd.DataFrame(X_test, columns=columns_name)
        return X_train, y_train, X_test, y_test
    X_train, y_train, X_test, y_test = data_split(X, y)
    X_train, y_train, X_val, y_val = data_split(X_train.values, y_train)
    y_test_df = pd.DataFrame(y_test)
    # Prepare dataset: training data: X_train, label: y_train
    train = lgb.Dataset(X_train, label=y_train)
    valid = lgb.Dataset(X_val, label=y_val)
    # Set the parameters
    params = {'task': 'train',
              'boosting_type': 'gbdt',
              'objective': 'regression',
              'metric': 'rmse',
              'learning_rate': 0.1 }
    # Train the model
    model = lgb.train(params,
                      train,
                      valid_sets=valid,
                      num_boost_round=3000,
                      early_stopping_rounds=100)
    # Prediction
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    # Display actual values and predicted values
    df_pred = pd.DataFrame({'regression_y_test':y_test,'regression_y_pred':y_pred})
    # Calculate MSE (Mean Square Error)
    mse = mean_squared_error(y_test, y_pred)
    # Calculate RSME = √MSE
    rmse = np.sqrt(mse)
    # r2 : Calculate the coefficient of determination
    r2 = r2_score(y_test,y_pred)
    df_Df = pd.DataFrame({'regression_y_test_'+no:y_test,'regression_y_pred_'+no:y_pred,'RMSE_'+no:rmse,'R2_'+no:r2})
    df_Df.to_csv(r""+"./"+outname+no+'.csv', encoding = 'shift-jis')
    importance = pd.DataFrame(model.feature_importance(importance_type='gain'), columns=['importance'])
    column_list=merge_data.drop(["target"], axis=1)
    importance["columns"] =list(column_list.columns)
    return importance
# Find out Top 50 features procedure / Run the model once
importance = reg_top10_lightGBM(training_validation_sets,"check_data","_1",1)

# Create a function that sorts and stores the values of feature importance.
def after_imp_save_sort(importance,outname,no):
    importance.sort_values(by='importance',ascending=False)
    i_df=importance.sort_values(by='importance',ascending=False)
    top50=i_df.iloc[0:51,:]
    g_dpc_pre= g_dpc_r_1.drop(["ID"], axis=1)
    g_dpc_Remove_duplicates=g_dpc_pre.drop_duplicates()
    g_dpc_r_columns=g_dpc_Remove_duplicates.rename(columns={'code':"columns"})
    importance_name=pd.merge(top50,g_dpc_r_columns)
    importance_all=pd.merge(i_df,g_dpc_r_columns)
    importance_all.to_csv(r""+"./"+outname+no+'importance_name_all'+'.csv', encoding = 'shift-jis')
    return importance_all
# Run a function to sort and save the values of feature importance.
top50_importance_all = after_imp_save_sort(importance,"check_data","_1")
# 10 runs of this procedure
dict = {}
for num in range(10):
    print(num+1)
    importance = reg_top10_lightGBM(training_validation_sets,"check_data","_"+str(num+1),num+1)
    top50_importance_all = after_imp_save_sort(importance,"check_data","_"+str(num+1))
    dict[str(num)] = top50_importance_all

# Recall and merge the saved CSV files
def concat_importance(First_pd,Next_pd):
    importance_1=pd.DataFrame(dict[First_pd])
    importance_1d=importance_1.drop_duplicates(subset='columns')
    importance_2=pd.DataFrame(dict[Next_pd])
    importance_2d=importance_2.drop_duplicates(subset='columns')
    importance_1_2=pd.concat([importance_1d, importance_2d])
    return importance_1_2
importance_1_2 = concat_importance("0","1")
importance_3_4 = concat_importance("2","3")
importance_5_6 = concat_importance("4","5")
importance_7_8 = concat_importance("6","7")
importance_9_10 = concat_importance("8","9")
importance_1_4=pd.concat([importance_1_2, importance_3_4])
importance_1_6=pd.concat([importance_1_4, importance_5_6])
importance_1_8=pd.concat([importance_1_6, importance_7_8])
importance_1_10=pd.concat([importance_1_8, importance_9_10])
# Calculate the total value of the feature importance for each code
group_sum=importance_1_10.groupby(["columns"]).sum()
group_sum_s = group_sum.sort_values('importance', ascending=False)
importance_group_sum=group_sum_s.reset_index()
# Create train/validation test data with all features
merge_data_test=pd.concat([training_validation_sets, test_df])
# Make features in the order of highest total feature impotance value
importance_top50_previous_data=importance_group_sum["columns"]
importance_top50_previous_data

# refine the data to top 50 features
dict_top50 = {}
pycaret_dict_top50 = {}

X = range(1, 51)
for i,v in enumerate(X):
    dict_top50[str(i)] = importance_top50_previous_data.iloc[v]
    pycaret_dict_top50[importance_top50_previous_data[i]] = merge_data_test[dict_top50[str(i)]]
pycaret_df_dict_top50=pd.DataFrame(pycaret_dict_top50)
# Add the value of target (: objective variable)
target_data=merge_data_test["target"]
target_top50_dataframe=pd.concat([target_data, pycaret_df_dict_top50], axis=1)
# adjust pandas (pycaret needs to set “str” to “int”)
target_top50_dataframe_int=target_top50_dataframe.astype('int')
target_top50_dataframe_columns=target_top50_dataframe_int.columns.astype(str)
numpy_target_top50=target_top50_dataframe_int.to_numpy()
target_top50_dataframe_pycaret=pd.DataFrame(numpy_target_top50,columns=target_top50_dataframe_columns)

# compare the models
from pycaret.classification import *
clf1 = setup(target_top50_dataframe_pycaret, target ='target',train_size = 0.8,data_split_shuffle=False,fold=10,session_id=0)
best_model = compare_models()
