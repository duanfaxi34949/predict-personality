# -*- coding: utf-8 -*-
"""此5*10次交叉验证,及每次内部5折交叉验证,代码"""
import datetime
import json
import numpy as np
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
# 采集数据量报警



CRAWL_WARRING_TOKEN = 'https://oapi.dingtalk.com/robot/send?access_token' \
                      '=4468d73265d5a0b7aedc0823e9692fb5b83e85d5deb48250bf7b54f0f27b1d17'

def dingmessage(token, text=''):
    # 请求的URL，WebHook地址
    url = token
    # 构建请求头部
    header = {"Content-Type": "application/json", "Charset": "UTF-8"}
    text = '时间:' + str(datetime.datetime.now()).split('.')[0] + '\r\n' + text
    # 构建请求数据
    message = {

        "msgtype": "text", "text": {"content": text}, "at": {

            "isAtAll": True}

    }
    # 对请求的数据进行json封装
    message_json = json.dumps(message)
    # 发送请求
    info = requests.post(url=url, data=message_json, headers=header, verify=False)
    # 打印返回的结果
    return info.text

def calculate_shap_values(model, X_train):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    return shap_values

def best(x, y, z,d):
    # 定义内部和外部交叉验证

    inner_cv = KFold(n_splits=5, shuffle=True, random_state=None)
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=None)

    # 设置模型的参数网格
    param_grid_enet = {
        'alpha': [ 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
                   0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
                   0.48, 0.49, 0.5, 0.51, 0.51, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64,
                   0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81,
                   0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
                   0.99, 1, 1.1, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7,
                   1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 4, 5, 10],
        'l1_ratio': [0.000001,0.00001, 0.0001, 0.01, 0.02, 0.03, 0.04, 0.045, 0.046, 0.047, 0.048, 0.049, 0.05, 0.06, 0.07, 0.08, 0.09,
                    0.1, 0.11, 0.12,0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
                    0.28, 0.29, 0.3, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49,
                    0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 0.9, 1]
    }
    param_grid_rf = {
        'n_estimators': [300,400,500],
        'max_depth': [None,10,20]
    }

    # 初始化模型
    enet = ElasticNet(random_state=42)
    rf = RandomForestRegressor(random_state=42)

    # 初始化网格搜索
    grid_search_enet = GridSearchCV(enet, param_grid_enet, cv=inner_cv, n_jobs=-1, scoring='r2')
    grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=inner_cv, n_jobs=-1, scoring='r2')

    # 存储结果
    all_performances = {
        'ElasticNet': [],
        'RandomForest': []
    }
    best_params_rf = None
    best_performance_rf = -float('inf')
    best_performance_enet = -float('inf')
    with open(f'{z}_best_params.txt', 'w') as ff:
        for _ in range(5):  # 重复5次交叉验证
            print(_)

            for train_idx, test_idx in outer_cv.split(x):
                X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # ElasticNet
                grid_search_enet.fit(X_train, y_train)
                y_pred_enet = grid_search_enet.best_estimator_.predict(X_test)
                r2_enet = r2_score(y_test, y_pred_enet),
                print(r2_enet)

                all_performances['ElasticNet'].append([
                    mean_squared_error(y_test, y_pred_enet),
                    np.sqrt(mean_squared_error(y_test, y_pred_enet)),
                    r2_enet[0],
                    mean_absolute_error(y_test, y_pred_enet),
                    pearsonr(y_test, y_pred_enet)[0],
                ])
                if r2_enet[0] > best_performance_enet:
                    best_performance_enet = r2_enet
                    best_params_enet = grid_search_enet.best_params_


                # RandomForest
                grid_search_rf.fit(X_train, y_train)
                best_rf = grid_search_rf.best_estimator_
                y_pred_rf = best_rf.predict(X_test)
                r2_rf = r2_score(y_test, y_pred_rf)
                print(r2_rf)

                all_performances['RandomForest'].append([
                    mean_squared_error(y_test, y_pred_rf),
                    np.sqrt(mean_squared_error(y_test, y_pred_rf)),
                    r2_rf,
                    mean_absolute_error(y_test, y_pred_rf),
                    pearsonr(y_test, y_pred_rf)[0],
                ])
                # 更新最优参数
                if r2_rf > best_performance_rf:
                    best_performance_rf = r2_rf
                    best_params_rf = grid_search_rf.best_params_

        ff.write(f'{z}弹性网络最优参数{best_params_enet}\n{z}随机森林模型最优参数{best_params_rf}')

    # 使用最优参数在整个数据集上训练随机森林模型
    best_rf_model = RandomForestRegressor(**best_params_rf, random_state=42)
    best_rf_model.fit(x, y)

    # 计算SHAP值
    explainer = shap.TreeExplainer(best_rf_model)
    shap_values = explainer.shap_values(x)
    # 获取特征的平均绝对SHAP值
    shap_sum = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': x.columns,
        'shap_importance': shap_sum
    })
    with open(f'{z}_total.txt', 'w') as file:
    # 对特征按SHAP值进行排序，并保存前20个
        top20_features = feature_importance_df.sort_values(by='shap_importance', ascending=False).head(20)
        top20_features_list = top20_features['feature'].tolist()
        file.writelines(str(top20_features_list))
        # 绘制SHAP值图表
        shap.summary_plot(shap_values, x, plot_type="dot", show=False)
        plt.savefig(f"{z}_RandomForest_shap_plot.png")
        plt.savefig(f"{z}_RandomForest_shap_plot.svg", dpi=300, format="svg")


    # 保存结果

        for model in ['ElasticNet', 'RandomForest']:
            performance_df = pd.DataFrame(all_performances[model], columns=['MSE', 'RMSE', 'R2', 'MAE', 'Correlation', ])
            # print(performance_df)
            mean_performance = performance_df.mean().to_dict()
            median_performance = performance_df.median().to_dict
            conf_interval_performance = performance_df.quantile([0.025, 0.975]).to_dict()

            file.write(f'\n{z}{model}\nMean Performance:{mean_performance}\nMedian Performance:{median_performance}\n95% CI Performance:{conf_interval_performance}\n')
            performance_df.to_csv(f"{z}{model}_50_results_performance.csv")



data = pd.read_csv('./only_tizhi.csv')
x1 = data.iloc[:, 0:60]     #前闭后开。索引60不包含
drop_index = ['17','46','47','48','50','52','53','55']
x2 = x1.drop(columns = drop_index,axis=1)
y2 = data.iloc[:,211]
best(x2,y2,'211','删除特定指定题目')






