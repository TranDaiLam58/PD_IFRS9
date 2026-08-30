import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, RocCurveDisplay

#define index outliers
def iqr_outlier(df):
    index_out = []
    for i in df.columns:
        sort = df[i].sort_values()
        q1 = sort.quantile(0.25)
        q3 = sort.quantile(0.75)
        iqr = q3-q1 
        upper = q3 + 1.5*iqr
        lower = q1 - 1.5*iqr
        index_lower = np.where(sort <=lower)[0]
        index_upper= np.where(sort >=upper)[0]
        index = np.concatenate((index_upper,index_lower), axis=0)
        index_out =  np.concatenate((index,index_out), axis=0)
    return index_out

def Z_score_outlier(df):
    outlier = set()
    for i in df.columns:
        Z_score = (df[i] - df[i].mean())/df[i].std()
        index = np.where(abs(Z_score) > 3)[0]
        outlier.update(index)
    return list(outlier)
        
# plot Heat Map
def heat_map(df, figsize, cmap):
    m = np.triu(np.ones_like(df.corr(),dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), mask=m, cmap=cmap, annot=True)
    plt.title('Heatmap decribe correlation features')
    plt.show()
    
# check IV
# def check_var_iv(bin,threshold):
#     mydict = {}
#     list_var = []
#     for key, value in bin.items():
#         if bin[key]['total_iv'][0] <= threshold:
#             mydict = {key:bin[key]['total_iv'][0]}
#             list_var.append(mydict)
#     return list_var   
def check_var_iv(bin):
    IV = pd.DataFrame(columns=['Features','Total_iv'])
    Features = []
    Total_iv = []
    for key, value in bin.items():
        #if bin[key]['total_iv'][0] <= 0.02:
            Features.append(key) #{key:bin[key]['total_iv'][0]}
            Total_iv.append(bin[key]['total_iv'][0])
    IV['Features'] = Features
    IV['Total_iv'] = Total_iv
    return IV.sort_values(ascending=False, by='Total_iv')

# display variance_inflation_factor
def display_vif(df_woe):
    vif_data = pd.DataFrame()
    vif_data['feature'] = df_woe.columns
    vif_data['VIF'] = [variance_inflation_factor(df_woe.values, i) for i in range(len(df_woe.columns))]
    return vif_data.sort_values(ascending=False, by='VIF')

# define features low correlation
def define_low_correlation(df_woe, target, threshold):
    corr = df_woe.corr()
    c = corr[target]
    low_corr = []
    for i in c.index:
        if c[i]<= threshold:
            low_corr.append(i)
    return low_corr

def check_multicollinearity_iv(df, target, bins, threshold):
    df_corr = df.corr()
    multicollinearity = pd.DataFrame()
    var1 = []
    var2 = []
    iv_var1 = []
    iv_var2 = []
    corr = []
    for i in df_corr.columns.drop(target):
        for j in df_corr.columns.drop(target):
            if abs(df_corr[i][j])>= threshold and i!=j:
                #if abs(df_corr[target][i]) > abs(df_corr[target][j]):
                var1.append(i)
                var2.append(j)
                iv_var1.append(bins[i[:-4]]['total_iv'][0])
                iv_var2.append(bins[j[:-4]]['total_iv'][0])
                corr.append(df_corr[i][j])
    multicollinearity['var1'] = var1
    multicollinearity['var2'] = var2
    multicollinearity['iv_var1'] = iv_var1
    multicollinearity['iv_var2'] = iv_var2
    multicollinearity['corr'] = corr
    return multicollinearity

def check_multicollinearity(df, target, bins, threshold):
    df_corr = df.corr()
    multicollinearity = pd.DataFrame()
    var1 = []
    var2 = []
    iv_var1 = []
    iv_var2 = []
    corr = []
    for i in df_corr.columns.drop(target):
        for j in df_corr.columns.drop(target):
            if abs(df_corr[i][j])>= threshold and i!=j:
                #if abs(df_corr[target][i]) > abs(df_corr[target][j]):
                var1.append(i)
                var2.append(j)
                iv_var1.append(bins[i[:-4]]['total_iv'][0])
                iv_var2.append(bins[j[:-4]]['total_iv'][0])
                corr.append(df_corr[i][j])
    multicollinearity['var1'] = var1
    multicollinearity['var2'] = var2
    multicollinearity['iv_var1'] = iv_var1
    multicollinearity['iv_var2'] = iv_var2
    multicollinearity['corr'] = corr
    return multicollinearity

def check_multicollinearity2(df, target, threshold):
    df_corr = df.corr()
    multicollinearity = pd.DataFrame()
    var1 = []
    var2 = []
    corr_var1 = []
    corr_var2 = []
    corr = []
    for i in df_corr.columns.drop(target):
        for j in df_corr.columns.drop(target):
            if abs(df_corr[i][j])>= threshold and i!=j:
                if abs(df_corr[target][i]) > abs(df_corr[target][j]):
                    var1.append(i)
                    var2.append(j)
                    corr_var1.append(df_corr[target][i])
                    corr_var2.append(df_corr[target][j])
                    corr.append(df_corr[i][j])
    multicollinearity['var1'] = var1
    multicollinearity['var2'] = var2
    multicollinearity['iv_var1'] = corr_var1
    multicollinearity['iv_var2'] = corr_var2
    multicollinearity['corr'] = corr
    return multicollinearity

#Define Highper Parameters LogisticRegression
# from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
# from sklearn.linear_model import LogisticRegression
# def Define_Highper_Parameters_LogisticRegression(X_train, y_train):
#     model = LogisticRegression()
#     solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#     penalty = ['l2','l1']
#     c_values = [100, 10, 1, 0.1, 0.001, 0.0001]
#     grid = dict(solver = solvers, penalty = penalty, C = c_values)
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=11)
#     grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
#     grid_result = grid_search.fit(X_train, y_train)
#     print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

#Define β,  α
def Lnodds(features, model):
    LR = pd.DataFrame()
    LR['Features']= features
    LR['β'] = model.coef_.tolist()[0]
    LR['α'] = [model.intercept_[0]]*len(features) 
    return LR

#draw ROC_curve
def draw_ROC_curve(y_real, y_score, figsize):
    plt.figure(figsize=figsize)
    fpr, tpr, thresholds = roc_curve(y_real, y_score)
    roc_au_s = roc_auc_score(y_real, y_score)
    plt.plot(fpr,tpr, label="Model AUC = {%.2f}" %roc_au_s)
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.title("ROC_curve", fontweight = 'bold')
    plt.xlabel("False positive rates")
    plt.ylabel("True positive rates")
    plt.legend()
    plt.show()
 
def draw_confusion_matrix(y_test, y_pred, cmap, figsize, xticklabels,yticklabels):  
    ''' xticklabels = [positive, negative] 
        yticklabels = [positive, negative]'''
    plt.figure(figsize=figsize) 
    sns.heatmap(confusion_matrix(y_test, y_pred), cmap = cmap, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, fmt='d').set_title('Confusion Matrix', fontweight = 'bold')
    plt.xlabel('Predict')
    plt.ylabel('Actual')
    
#Check best fit model
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
# from sklearn import model_selection

# Models = {
#     "Logistic": LogisticRegression(),
#     "SVC": SVC(),
#     "DecisionTree": DecisionTreeClassifier(),
#     "RandomForest": RandomForestClassifier(),
#     "GaussianNaiveBayes": GaussianNB()    
# }
# cv_results = pd.DataFrame(columns=['model', 'train_score', 'test_score'])
# for key in Models.keys():
#     cv_res = model_selection.cross_validate(Models[key], X_train, y_train, 
#                                              return_train_score=True,
#                                              scoring="f1",
#                                              cv=5, n_jobs=-1)
#     res = {
#         'model': key, 
#         'train_score': cv_res["train_score"].mean(), 
#         'test_score': cv_res["test_score"].mean(),
#         'fit_time': cv_res["fit_time"].mean(),
#         'score_time': cv_res["score_time"].mean(),
#         }
#     cv_results = cv_results.append(res, ignore_index=True)
#     print("CV for model:", key, "done.")
# cv_results

# define Gini
def Gini(y_true, y_predict):
    assert y_true.shape == y_predict.shape
    n_samples = y_true.shape[0]
    arr = np.array([y_true, y_predict]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    L_true = np.cumsum(true_order)/np.sum(true_order)
    L_pred = np.cumsum(pred_order)/np.sum(pred_order)
    L_ones = np.linspace(1/n_samples, 1, n_samples)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    return G_pred/G_true

# plt.figure(figsize=(9, 6))
# def gini_plot(array):
#     array = sorted(array)
#     x = [i/len(array) for i in range(len(array))]
#     y = [sum(array[0:i+1])/sum(array) for i in range(len(array))]
#     return x,y
# x,y = gini_plot(y_pred_proba)
# plt.plot(x, y, label = "Lorenz Curve")
# plt.plot([0,1], [0,1], label = "Ideal Curve")
# plt.legend(loc="upper left")
# plt.show()

# value = df['BAD'].value_counts()
# fig, axs = plt.subplots(1,2, figsize = (12, 6))
# palette = sns.color_palette("bright", len(value))
# axs[0].pie(value, labels = value.index, autopct = '%1.1f%%')
# axs[0].set_title('Distribution of BAD', fontweight = 'bold')
# sns.barplot(x = value.index, y = value.values, ax=axs[1])
# axs[1].set_title('Count of BAD', fontweight = 'bold')
# plt.show()

# model_importance = LogisticRegression()
# model_importance.fit(X_train, y_train)
# importance = model_importance.coef_[0]
# feature_importances = pd.DataFrame()
# feature_importances['feature'] = X_train.columns
# feature_importances['importance'] = importance
# feature_importances.sort_values(by='importance', ascending=False)

# AUC = roc_auc_score(y_score=y_pred_pro_test,y_true=y_test)
# Gini = 2*AUC - 1

# data_score_x=data_score[(data_score.LOAN_STATUS ==1)]
# data_score_y=data_score[(data_score.LOAN_STATUS ==0)]
# # Plotting the KDE Plot
# plt.figure(figsize=(20,15))
# sns.kdeplot(data_score.loc[(data_score['LOAN_STATUS']==1),
#             'Score'], color='r',shade=True,label ="BAD")
# sns.kdeplot(data_score.loc[(data_score['LOAN_STATUS']==0),
#             'Score'], color='b',shade=True,legend=True,label ="GOOD") 
# # Setting the X and Y Label
# plt.xlabel('Score')
# plt.ylabel('Probability Density')
# plt.legend(title='Distribution score', loc='upper left', labels=['BAD', 'GOOD'])
# plt.show()

# name = model.feature_names_in_
# coef = model.coef_[0]
# intercept = model.intercept_[0]

# result = 'Ln(odds) = '
# for i in range(len(name)):
#     if coef[i]<0:
#         result = result + ' ' + str(round(coef[i],5)) + '*' + name[i]
#     else:
#         result = result + ' + ' + str(round(coef[i],5)) + '*' + name[i]
# if intercept<0: 
#     result = result + ' ' + str(intercept)
# else:
#     result = result + ' + ' + str(intercept)
# print(result)

# plt.figure(figsize=(20,15))

# num_cols = 4
# num_rows = (len(df_corr.columns) + num_cols - 1) // num_cols  # Tính số hàng cần thiết

# for i, column in enumerate(df_corr.columns):
#     plt.subplot(num_rows, num_cols, i + 1)  # Số hàng, số cột
#     sns.boxplot(data = df_corr, y= column)
#     plt.title(column)
# plt.tight_layout()
# plt.show()