import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.simplefilter('ignore')
# OUR = int(input("OUR:"))
# num = float(input("Linear threshold（0-1）："))
pp = pd.read_csv("./data.csv",index_col=0)
pp_test = pd.read_csv("./data_test.csv",index_col=0)
pp = pd.DataFrame(pp)
pp_test = pd.DataFrame(pp_test)
# # print(pp)
V = float(input("Volume:"))  #体积
O = float(input("Oil concentration:")) # c1 菜籽油浓度
C = float(input("Control oil concentration:")) # c2  控制的菜籽油浓度
O1 = float(input("glu concentration:")) # c3  糖浓度
C1 = float(input("Control glu concentration:")) # c4  控制的糖浓度
oil_model_score = {}    #菜籽油模型得分
glu_model_score = {}    #糖模型得分
pro_model_score = {}    #
# print(model_score)
# LR
print(" ")
print("LR")  #使用线性模型
LR = linear_model.LinearRegression()  #线性模型
LR_glu = linear_model.LinearRegression()
LR_pro = linear_model.LinearRegression()
LRM = linear_model.LinearRegression()
LRM_glu = linear_model.LinearRegression()
LRM_pro = linear_model.LinearRegression()
LR_oil_n = {}   #油模型得分
LR_glu_n = {}   #糖模型得分
LR_pro_n = {}   #产率模型得分
for i in range(0,7):
    LR.fit(pp.iloc[:,i].values.reshape(-1,1), pp.iloc[:,9:10]) # 数据拟合
    LR_oil_score = LR.score(pp.iloc[:,i].values.reshape(-1,1), pp.iloc[:,9:10])  # 模型得分
    LR_oil_n[pp.columns[i] + pp.columns[9]] = LR_oil_score  # 模型得分
LR.fit(pp.iloc[:,0:1], pp.iloc[:,9:10])
LR_oil_n = pd.DataFrame(LR_oil_n,index=[0]).T
LR_max = LR_oil_n.sort_values(by=0,ascending=False).iloc[0]  #线性模型得分排序，求最大值
print(LR_max.name,LR_max[0])
oil_model_score ["LR"] = LR_max[0]
# print(LR_oil_n)
print(" ")

 # glu
for i in range(0, 7):
    LR_glu.fit(pp.iloc[:, i].values.reshape(-1, 1), pp.iloc[:, 8:9])
    LR_glu_score = LR_glu.score(pp.iloc[:, i].values.reshape(-1, 1), pp.iloc[:, 8:9])
    LR_glu_n[pp.columns[i] + pp.columns[8]] = LR_glu_score
LR_glu.fit(pp.iloc[:,0:1], pp.iloc[:,8:9])
LR_glu_n = pd.DataFrame(LR_glu_n,index=[0]).T
LR_glu_max = LR_glu_n.sort_values(by=0,ascending=False).iloc[0]
print(LR_glu_max.name,LR_glu_max[0])
glu_model_score ["LR"] = LR_glu_max[0]
# print(LR_glu_n)
print(" ")

# # productivity
for i in range(0,7):
    LR_pro.fit(pp.iloc[:, i].values.reshape(-1, 1), pp.iloc[:, 7:8])
    LR_pro_score = LR_pro.score(pp.iloc[:, i].values.reshape(-1, 1), pp.iloc[:, 7:8])
    LR_pro_n[pp.columns[i] + pp.columns[7]] = LR_pro_score
LR_pro.fit(pp.iloc[:, 0:1], pp.iloc[:,7:8])
LR_pro_n = pd.DataFrame(LR_pro_n, index=[0]).T
LR_pro_max = LR_pro_n.sort_values(by=0, ascending=False).iloc[0]
print(LR_pro_max.name, LR_pro_max[0])
oil_model_score["LR"] = LR_pro_max[0]
# print(LR_pro_n)

    # LR_oil["score"] = LR_oil.append(LR_score)
    # print(pp.columns[i],pp.columns[j])
    # print(LR_score)
    # if LR_score >= 0.9:
    #     print(pp.columns[i],pp.columns[j])
    #     print(LR_score)

print(" ")
print("LRM") #使用多元线性回归模型
LRM.fit(pp.iloc[:, :7], pp.iloc[:, 9:10]) # oil
LRM_score = LRM.score(pp.iloc[:, :7], pp.iloc[:, 9:10])

LRM_glu.fit(pp.iloc[:, :7], pp.iloc[:, 8:9]) # glu
LRM_glu_score = LRM_glu.score(pp.iloc[:, :7], pp.iloc[:, 8:9])

LRM_pro.fit(pp.iloc[:, :7], pp.iloc[:, 7:8]) # pro
LRM_pro_score = LRM_pro.score(pp.iloc[:, :7], pp.iloc[:, 7:8])
print("oil score:{}".format(LRM_score))
print("glu score:{}".format(LRM_glu_score))
print("pro score:{}".format(LRM_pro_score))
oil_model_score["LRM"] = LRM_score
glu_model_score["LRM"] = LRM_glu_score
pro_model_score["LRM"] = LRM_pro_score

print(" ")
print("SVM——linear")  #使用支持向量机（线性）
svm_oil = svm.SVR(kernel="linear",C=0.05)
svm_glu = svm.SVR(kernel="linear",C=0.05)
svm_pro = svm.SVR(kernel="linear",C=0.05)
svm_oil.fit(pp.iloc[:, :7], pp.iloc[:, 9:10]) # oil
svm_score = svm_oil.score(pp.iloc[:, :7], pp.iloc[:, 9:10])
svm_glu.fit(pp.iloc[:, :7], pp.iloc[:, 8:9]) # glu
svm_glu_score = svm_glu.score(pp.iloc[:, :7], pp.iloc[:, 8:9])
svm_pro.fit(pp.iloc[:, :7], pp.iloc[:, 7:8]) # pro
svm_pro_score = svm_pro.score(pp.iloc[:, :7], pp.iloc[:, 7:8])
print("oil score:{}".format(svm_score))
print("glu score:{}".format(svm_glu_score))
print("pro score:{}".format(svm_pro_score))
oil_model_score["svm"] = svm_score
glu_model_score["svm"] = svm_glu_score
pro_model_score["svm"] = svm_pro_score

print(" ")
print("pls")  #使用偏最小二乘算法
pls = PLSRegression()
pls_glu = PLSRegression()
pls_pro = PLSRegression()
pls.fit(pp.iloc[:, :7], pp.iloc[:, 9:10])
pls_score = pls.score(pp.iloc[:, :7], pp.iloc[:, 9:10])
pls_glu.fit(pp.iloc[:, :7], pp.iloc[:, 8:9]) # glu
pls_glu_score = pls_glu.score(pp.iloc[:, :7], pp.iloc[:, 8:9])
pls_pro.fit(pp.iloc[:, :7], pp.iloc[:, 7:8]) # pro
pls_pro_score = pls_pro.score(pp.iloc[:, :7], pp.iloc[:, 7:8])
print("oil score:{}".format(pls_score))
print("glu score:{}".format(pls_glu_score))
print("pro score:{}".format(pls_pro_score))
oil_model_score["pls"] = pls_score
glu_model_score["pls"] = pls_glu_score
pro_model_score["pls"] = pls_pro_score

print("")
print("RF")  #使用随机森林模型
RF = RandomForestRegressor()
RF_glu = RandomForestRegressor()
RF_pro = RandomForestRegressor()
RF.fit(pp.iloc[:, :7], pp.iloc[:, 9:10]) # oil
RF_score = RF.score(pp.iloc[:, :7], pp.iloc[:, 9:10])
RF_glu.fit(pp.iloc[:, :7], pp.iloc[:, 8:9]) # glu
RF_glu_score = RF_glu.score(pp.iloc[:, :7], pp.iloc[:, 8:9])
RF_pro.fit(pp.iloc[:, :7], pp.iloc[:, 7:8]) # pro
RF_pro_score = RF_pro.score(pp.iloc[:, :7], pp.iloc[:, 7:8])
print("oil score:{}".format(RF_score))
print("glu score:{}".format(RF_glu_score))
print("pro score:{}".format(RF_pro_score))
oil_model_score["RF"] = RF_score
glu_model_score["RF"] = RF_glu_score
pro_model_score["RF"] = RF_pro_score

print(" ")
print("gbr") #使用梯度提升算法
gbr = GradientBoostingRegressor()
gbr_glu = GradientBoostingRegressor()
gbr_pro = GradientBoostingRegressor()
gbr.fit(pp.iloc[:, :7], pp.iloc[:, 9:10]) # oil
gbr_score = gbr.score(pp.iloc[:, :7], pp.iloc[:, 9:10])
gbr_glu.fit(pp.iloc[:, :7], pp.iloc[:, 8:9]) # glu
gbr_glu_score = gbr_glu.score(pp.iloc[:, :7], pp.iloc[:, 8:9])
gbr_pro.fit(pp.iloc[:, :7], pp.iloc[:, 7:8]) # pro
gbr_pro_score = gbr_pro.score(pp.iloc[:, :7], pp.iloc[:, 7:8])
print("oil score:{}".format(gbr_score))
print("glu score:{}".format(gbr_glu_score))
print("pro score:{}".format(gbr_pro_score))
oil_model_score["gbr"] = gbr_score
glu_model_score["gbr"] = gbr_glu_score
pro_model_score["gbr"] = gbr_pro_score

# 模型得分排序
oil_model_score = pd.DataFrame(oil_model_score,index=["score"]).T.sort_values(by="score",ascending=False)
# oil_model_score = oil_model_score
glu_model_score = pd.DataFrame(glu_model_score,index=["score"]).T.sort_values(by="score",ascending=False)
# glu_model_score = glu_model_score
pro_model_score = pd.DataFrame(pro_model_score,index=["score"]).T.sort_values(by="score",ascending=False)
# pro_model_score = pro_model_score
name = np.array(oil_model_score.iloc[0].name)   #菜籽油得分最高的模型算法
name1 = np.array(glu_model_score.iloc[0].name)  #葡萄糖得分最高的模型算法
name2 = np.array(pro_model_score.iloc[0].name)  #生产率得分最高的模型算法
# name = "gbr"
# name1 = "gbr"
# name2 = "gbr"
print(" ")
print("oil Optimal model：{}".format(name))
print("glu Optimal model：{}".format(name1))
print("pro Optimal model：{}".format(name2))

# 使用验证集数据进行验证
print(" ")
LR_test = pp_test.iloc[-1,:]
test = np.array(pp_test.iloc[-1, :7])
if name == "LR":
    s_oil1 = LR.predict(LR_test[0].reshape(1, -1))
    pp = pp.append(LR_test,ignore_index=True)
    pp.to_csv("./data.csv",)
    # print(s_oil1)
    s_oil2 = C * V - O * V
    s = s_oil1 + s_oil2
    print("fed oil：{}".format(s))
elif name == "LRM":
    s_oil1 = LRM.predict(test.reshape(1, -1))
    pp = pp.append(pp_test.iloc[-1, :], ignore_index=True)
    pp.to_csv("./data.csv", )
    # print(s_oil1)
    s_oil2 = C * V - O * V
    s = s_oil1 + s_oil2
    print("fed oil：{}".format(s))
elif name == "svm":
    s_oil1 = svm_oil.predict(test.reshape(1, -1))
    pp = pp.append(pp_test.iloc[-1, :], ignore_index=True)
    pp.to_csv("./data.csv", )
    # print(s_oil1)
    s_oil2 = C * V - O * V
    s = s_oil1 + s_oil2
    print("fed oil：{}".format(s))
elif name == "pls":
    s_oil1 = pls.predict(test.reshape(1, -1))
    pp = pp.append(pp_test.iloc[-1, :], ignore_index=True)
    pp.to_csv("./data.csv", )
    # print(s_oil1)
    s_oil2 = C * V - O * V
    s = s_oil1 + s_oil2
    print("fed oil：{}".format(s))
elif name == "RF":
    s_oil1 = RF.predict(test.reshape(1, -1))
    pp = pp.append(pp_test.iloc[-1, :], ignore_index=True)
    pp.to_csv("./data.csv", )
    # print(s_oil1)
    s_oil2 = C * V - O * V
    s = s_oil1 + s_oil2
    print("fed oil：{}".format(s))
elif name == "gbr":
    s_oil1 = gbr.predict(test.reshape(1, -1))
    pp = pp.append(pp_test.iloc[-1, :], ignore_index=True)
    pp.to_csv("./data.csv", )
    # print(s_oil1)
    s_oil2 = C * V - O * V
    s = s_oil1 + s_oil2
    print("fed oil：{}".format(s))

# glu
if name1 == "LR":
    s_glu1 = LR_glu.predict(LR_test[0].reshape(1, -1))
    # print(s_oil1)
    s_glu2 = C1 * V - O1 * V
    s = s_glu1 + s_glu2
    print("fed glucose：{}".format(s))
elif name1 == "LRM":
    s_glu1 = LRM_glu.predict(test.reshape(1, -1))
    # print(s_oil1)
    s_glu2 = C1 * V - O1 * V
    s = s_glu1 + s_glu2
    print("fed glucose：{}".format(s))
elif name1 == "svm":
    s_glu1 = svm_glu.predict(test.reshape(1, -1))
    # print(s_oil1)
    s_glu2 = C1 * V - O1 * V
    s = s_glu1 + s_glu2
    print("fed glucose：{}".format(s))
elif name1 == "pls":
    s_glu1 = pls_glu.predict(test.reshape(1, -1))
    # print(s_oil1)
    s_glu2 = C1 * V - O1 * V
    s = s_glu1 + s_glu2
    print("fed glucose：{}".format(s))
elif name1 == "RF":
    s_glu1 = RF_glu.predict(test.reshape(1, -1))
    # print(s_oil1)
    s_glu2 = C1 * V - O1 * V
    s = s_glu1 + s_glu2
    print("fed glucose：{}".format(s))
elif name1 == "gbr":
    s_glu1 = gbr_glu.predict(test.reshape(1, -1))
    # print(s_oil1)
    s_glu2 = C1 * V - O1 * V
    s = s_glu1 + s_glu2
    print("fed glucose：{}".format(s))

# pro
if name2 == "LR":
    s_pro = LR_pro.predict(LR_test[0].reshape(1, -1))
    print("productivity：{}".format(s_pro))
elif name2 == "LRM":
    s_pro = LRM_pro.predict(test.reshape(1, -1))
    print("productivity：{}".format(s_pro))
elif name2 == "svm":
    s_pro = svm_pro.predict(test.reshape(1, -1))
    print("productivity：{}".format(s_pro))
elif name2 == "pls":
    s_pro = pls_pro.predict(test.reshape(1, -1))
    print("productivity：{}".format(s_pro))
elif name2 == "RF":
    s_pro = RF_pro.predict(test.reshape(1, -1))
    print("productivity：{}".format(s_pro))
elif name2 == "gbr":
    s_pro = gbr_pro.predict(test.reshape(1, -1))
    print("productivity：{}".format(s_pro))
