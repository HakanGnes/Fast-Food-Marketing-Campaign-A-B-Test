# Fast Food Marketing Campaign A\B Test
# IBM Watson Analytics Marketing Campaign

# About Dataset
# Scenario
# A fast-food chain plans to add a new item to its menu.
# However, they are still undecided between three possible marketing campaigns
# for promoting the new product. In order to determine which promotion has the greatest effect on sales,
# the new item is introduced at locations in several randomly selected markets.
# A different promotion is used at each location, and the weekly sales of the new item are recorded
# for the first four weeks.
#
# Goal
# Evaluate A/B testing results and decide which marketing strategy works the best.
#
# Columns
# MarketID: unique identifier for market
# MarketSize: size of market area by sales
# LocationID: unique identifier for store location
# AgeOfStore: age of store in years
# Promotion: one of three promotions that were tested
# week: one of four weeks when the promotions were run
# SalesInThousands: sales amount for a specific LocationID, Promotion, and week
# Acknowledgements
# https://rpubs.com/ksdwivedy/finalRProject
# Hands-On Data Science for Marketing book by Yoon Hyup Hwang
# https://unsplash.com/@shaafi for the picture
# IBM Watson Analytics community for this dataset.

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#  Data Preparing

df_ = pd.read_csv("measurement_problems/datasets/WA_Marketing-Campaign.csv")
df = df_.copy()
df.head()


# Data Control
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# Outlier Check

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


outlier_thresholds(df, "SalesInThousands")
check_outlier(df, "SalesInThousands")

# HYPOTHESIS

""" 
HO : M1 = M2 = M3 : There is no a statistically difference in Promotions 
H1 : M1 != M2 != M3 : There is a statistically difference in Promotions 
"""

# Let's take a look at the averages of version usage.

df.groupby("Promotion").agg({"SalesInThousands": "mean"})

df.head()

# AB Testing (ANOVA)

# After checking the normality assumption and variance homogeneity,
# we will decide to apply a parametric or non-parametric test.

############################
# Normality Assumption
############################

# H0: Normal distribution assumption is provided.
# H1: The assumption of normal distribution is not provided.

test_stat, pvalue = shapiro(df.loc[df["Promotion"] == 3, "SalesInThousands"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.9208, p-value = 0.0000

test_stat, pvalue = shapiro(df.loc[df["Promotion"] == 2, "SalesInThousands"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.9145, p-value = 0.0000

test_stat, pvalue = shapiro(df.loc[df["Promotion"] == 1, "SalesInThousands"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 0.9153, p-value = 0.0000

promotions = df["Promotion"].unique()

for promotion in promotions:
    test_stat, p_value = shapiro(df[df["Promotion"] == promotion]["SalesInThousands"])

    print("Promotion : {} \t T-stat : {} \t P-value : {} \t Reject H_0: {}".format(promotion, test_stat, p_value,
                                                                                   p_value < 0.05))

" We reject H0,because p-value < 0.05, unless we should use non-parametric ANOVA Test"

############################
# Assumption of Variance Homogeneity
############################
# Actually, in this instance we do not need to check the homogeneity of variance.
# Because the assumption of normality was rejected.
# In this case automatically we should use non-parametric method.

# H0: Variances are Homogeneous
# H1: Variances Are Not Homogeneous

test_stat, pvalue = levene(df.loc[df["Promotion"] == 3, "SalesInThousands"],
                           df.loc[df["Promotion"] == 2, "SalesInThousands"],
                           df.loc[df["Promotion"] == 1, "SalesInThousands"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
print("P-value : {} \t H_0 reject : {}".format(pvalue, pvalue < 0.05))

# Test Stat = 1.2697, p-value = 0.2818

"We cannot reject H0,Variances are Homogeneous but we must use non-parametric way."

######################################################
# ANOVA (Analysis of Variance)
######################################################

test_stat, p_value = kruskal(df.loc[df["Promotion"] == 3, "SalesInThousands"],
                           df.loc[df["Promotion"] == 2, "SalesInThousands"],
                           df.loc[df["Promotion"] == 1, "SalesInThousands"])

print("P-value : {} \t H_0 reject : {}".format(p_value, p_value<0.05))

kruskal(df.loc[df["Promotion"] == 3, "SalesInThousands"],
                           df.loc[df["Promotion"] == 2, "SalesInThousands"],
                           df.loc[df["Promotion"] == 1, "SalesInThousands"])

# KruskalResult(statistic=53.29475169322799, pvalue=2.6741866266697816e-12)


"We reject the H0 meaning that at least one group differs in median from the others."

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["SalesInThousands"], df["Promotion"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

"We can see that There are a difference between 1 and 3 & 2 and 3."