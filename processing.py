import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

filename = './dataset/NFL Play by Play 2009-2017 (v4).csv'
with open(filename) as f:
    reader = csv.DictReader(f)
    zero=0
    one=0
    na=0
    for row in reader:
        if row['FirstDown'] == "0":
            zero+=1
        elif row['FirstDown'] == "1":
            one+=1
        elif row['FirstDown']=="NA":
            na+=1
    print "\n\n\nFirstDown"
    print "0:\t\t\t\t",zero,"\n1:\t\t\t\t",one,"\nNA:\t\t\t\t",na
#     new_li = []
#     count={}
#     for row in reader:
#         if row["Permit Type Definition"] not in new_li:
#             new_li.append(row["Permit Type Definition"])
#         count[row["Permit Type Definition"]] = count.setdefault(row["Permit Type Definition"], 0)
#         count[row["Permit Type Definition"]] += 1
#     print count



# import operator
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import scipy.stats as stats
# from collections import Counter
# from pandas import Series, DataFrame
#
# df=pd.DataFrame(pd.read_csv(r'./dataset/NFL Play by Play 2009-2017 (v4).csv',low_memory=False))
# fp_numerical = open(r'./NFL Play by Plays/result_NFL_numerical.txt','a+')
# fp_nominal = open(r'./NFL Play by Plays/result_NFL_nominal.txt','a+')
# attribute_all = df.columns
# attribute_numerical = []
# attribute_nominal = []
# for item in attribute_all:
#     if type(df[item][0]) == str or type(df[item][0]) == float:
# 	attribute_nominal.append(item)
#     else:
# 	attribute_numerical.append(item)
#
# print >> fp_numerical,df.describe()
# for item in attribute_nominal:
#     print >> fp_nominal,item,Counter(df[item])
#
# #save histogram
# fig = plt.figure(figsize = (40,30))
# i = 1
# for item in attribute_numerical:
#     if type(df[item][0]) <> str and type(df[item][0]) <> float :
#         ax = fig.add_subplot(8,8,i)
#         df[item].plot(kind = 'hist', title = item, ax = ax)
#         i += 1
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
# fig.savefig('./NFL Play by Plays/image/histogram.jpg')
#
# #save boxplot
# fig = plt.figure(figsize = (40,30))
# i = 1
# for item in attribute_numerical:
#     if type(df[item][0]) <> str and type(df[item][0]) <> float :
#         ax = fig.add_subplot(8,8,i)
#         df[item].plot(kind = 'box')
#         i += 1
# fig.savefig('./NFL Play by Plays/image/boxPlot.jpg')
#
#
# #save qqplot
# fig = plt.figure(figsize = (40,30))
# i = 1
# for item in attribute_numerical:
#     if type(df[item][0]) <> str and type(df[item][0]) <> float :
#         ax = fig.add_subplot(8,8,i)
#         sm.qqplot(df[item], ax = ax)
#         ax.set_title(item)
#         i += 1
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
# fig.savefig('./NFL Play by Plays/image/qqPlot.jpg')
#
#
# # find null list
# nan_list = pd.isnull(df[attribute_numerical]).any(1).nonzero()[0]
#
# # use dropna() delete null
# data_filtrated = df.dropna()
#
# fig = plt.figure(figsize = (40,30))
#
# i = 1
#
# # for numerical,save histogram
# for item in attribute_numerical:
#     ax = fig.add_subplot(8,8,i)
#     ax.set_title(item)
#     ax.axvline(df[item].mean(), color = 'r')
#     ax.axvline(data_filtrated[item].mean(), color = 'b')
#     df[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
#     if not data_filtrated[item].empty:
#         data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'filtrated', legend = True)
#     i += 1
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
#
# # save image and data
# fig.savefig('./NFL Play by Plays/image/missing_data_delete.jpg')
# data_filtrated.to_csv('./data_output/missing_data_delete.csv', mode = 'w', encoding='utf-8', index = False,header = False)
#
#
#
# # use the most data fill null
# data_filtrated = df.copy()
#
# for item in attribute_numerical:
#     # find out the most data
#     most_frequent_value = data_filtrated[item].value_counts().idxmax()
#     # fill null
#     data_filtrated[item].fillna(value = most_frequent_value, inplace = True)
#
#
# fig = plt.figure(figsize = (40,30))
#
# i = 1
#
# # for numerical,save histogram
# for item in attribute_numerical:
#     if type(df[item][0]) <> str and type(df[item][0]) <> float and (not df[item].empty):
#         ax = fig.add_subplot(8,8,i)
#         ax.set_title(item)
# 	ax.axvline(df[item].mean(), color = 'r')
#         ax.axvline(data_filtrated[item].mean(), color = 'b')
#         df[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
# 	if not data_filtrated[item].empty:
#             data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
#         i += 1
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
#
# # save image and data
# fig.savefig('./NFL Play by Plays/image/missing_data_most.jpg')
# data_filtrated.to_csv('./data_output/missing_data_most.csv', mode = 'w', encoding='utf-8', index = False,header = False)
#
#
# # use the corelation between attribute fill null
# data_filtrated = df.copy()
#
# for item in attribute_numerical:
#     data_filtrated[item].interpolate(inplace = True)
#
#
# fig = plt.figure(figsize = (40,30))
#
# i = 1
#
# # for numerical,save histogram
# for item in attribute_numerical:
#     if type(df[item][0]) <> str and type(df[item][0]) <> float:
#         ax = fig.add_subplot(8,8,i)
#         ax.set_title(item)
# 	ax.axvline(df[item].mean(), color = 'r')
#         ax.axvline(data_filtrated[item].mean(), color = 'b')
#         df[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
# 	if not data_filtrated[item].empty:
#             data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
#         i += 1
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
#
# # save image and data
# fig.savefig('./NFL Play by Plays/image/missing_data_corelation.jpg')
# data_filtrated.to_csv('./data_output/missing_data_corelation.csv', mode = 'w', encoding='utf-8', index = False,header = False)
#
#
#
# # use the similarity between data fill null
# data_norm = df.copy()
#
# data_norm[attribute_numerical] = data_norm[attribute_numerical].fillna(0)
#
# data_norm[attribute_numerical] = data_norm[attribute_numerical].apply(lambda x : (x - np.mean(x)) / (np.max(x) - np.min(x)))
#
# score = {}
# range_length = len(df[attribute_numerical])
# for i in range(0, range_length):
#     score[i] = {}
#     for j in range(0, range_length):
#         score[i][j] = 0
#
#
# for i in range(0, range_length):
#     for j in range(i, range_length):
#         for item in attribute_numerical:
#             temp = abs(data_norm.iloc[i][item] - data_norm.iloc[j][item])
#             score[i][j] += temp
#         score[j][i] = score[i][j]
#
#
# data_filtrated = df.copy()
#
# for index in nan_list:
#     best_friend = sorted(score[index].items(), key=operator.itemgetter(1), reverse = False)[1][0]
#     for item in attribute_numerical:
#         if pd.isnull(data_filtrated.iloc[index][item]):
#             if pd.isnull(df.iloc[best_friend][item]):
#                 data_filtrated.ix[index, item] = df[item].value_counts().idxmax()
#             else:
#                 data_filtrated.ix[index, item] = df.iloc[best_friend][item]
#
# fig = plt.figure(figsize = (40,30))
#
# i = 1
#
# # for numerical,save histogram
# for item in attribute_numerical:
#     if type(df[item][0]) <> str and type(df[item][0]) <> float:
#         ax = fig.add_subplot(8,8,i)
#         ax.set_title(item)
# 	ax.axvline(df[item].mean(), color = 'r')
#         ax.axvline(data_filtrated[item].mean(), color = 'b')
#         df[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'origin', legend = True)
# 	if not data_filtrated[item].empty:
#             data_filtrated[item].plot(ax = ax, alpha = 0.5, kind = 'hist', label = 'droped', legend = True)
#         i += 1
# plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
#
# # save image and data
# fig.savefig('./NFL Play by Plays/image/missing_data_similarity.jpg')
# data_filtrated.to_csv('./data_output/missing_data_similarity.csv', mode = 'w', encoding='utf-8', index = False,header = False)
#
#
