import pickle as pkl
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
import itertools as it
import random
import numpy as np
from pyspark.ml.recommendation import ALS

import sys
import itertools

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import *

from operator import itemgetter
from itertools import groupby
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

spark = (SparkSession.builder
         .appName('train_als')
         .master('yarn')
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
'''
train_data_path='gs://aa7513/Train_data_two.parquet/part-00000-5dded405-4ce0-46e9-9a23-208f706a2010-c000.snappy.parquet'
val_data_path='gs://aa7513/Val_data_two.parquet/part-00000-621f87ab-4319-4492-a88c-3b0453a523bc-c000.snappy.parquet'
test_data_path='gs://aa7513/Test_data_two.parquet/part-00000-46bbfb1f-3358-481c-8022-f1fb66d027d5-c000.snappy.parquet'
'''


train_data_path='gs://aa7513/Train_data_five.parquet/part-00000-56c0e310-3b43-4001-9c21-ea46c0f26c81-c000.snappy.parquet'
val_data_path='gs://aa7513/Val_data_five.parquet/part-00000-87162e7e-a3d3-4e88-bf3d-fa897675136a-c000.snappy.parquet'
test_data_path='gs://aa7513/Test_data_five.parquet/part-00000-08307416-dae3-49d3-a3d4-07571c163978-c000.snappy.parquet'


train_data = spark.read.parquet(train_data_path)
val_data = spark.read.parquet(val_data_path) 
test_data= spark.read.parquet(test_data_path) 

train_data.createOrReplaceTempView('train_data')
val_data.createOrReplaceTempView('val_data')
test_data.createOrReplaceTempView('test_data')

val = spark.sql('select user_id as user, rating, book_id as item from val_data')
train = spark.sql('select user_id as user, rating, book_id as item from train_data')
test = spark.sql('select user_id as user, rating, book_id as item from test_data')

train.printSchema()
val.printSchema()
test.printSchema()

rank_  = [5,10,15,20,25,30,35,40]
regParam_ = [0.001, 0.01, 0.1, 1.0, 10.0]

user_id = val.select('user').distinct()                          
true_label = val.select('user', 'item').groupBy('user').agg(expr('collect_list(item) as true_item'))
train_label = train.select('user', 'item').groupBy('user').agg(expr('collect_list(item) as train_item'))
param_grid = it.product(rank_, regParam_)

map_scores=[]
ndcg_scores=[]
mpa_scores=[]

train.createOrReplaceTempView('train')
for i in param_grid:
    print('Start Training for {}'.format(i))
    als = ALS(rank = i[0], maxIter=10, regParam=i[1], coldStartStrategy="drop",seed=42)
    model = als.fit(train)
    print('Finish Training for {}'.format(i))
    res = model.recommendForUserSubset(user_id,500)
    prediction_label = res.select('user','recommendations.item')
    condition = [prediction_label.user == train_label.user, prediction_label.item == train_label.train_item]
    pred_filtered = prediction_label.join(F.broadcast(train_label), condition, 'left_anti').rdd.map(lambda row: (row[1], row[2]))
    pred_true_rdd = prediction_label.join(F.broadcast(true_label), 'user', 'inner').rdd.map(lambda row: (row[1], row[2]))
    print('Start Evaluating for {}'.format(i))
    metrics = RankingMetrics(pred_true_rdd)
    map_ = metrics.meanAveragePrecision
    ndcg = metrics.ndcgAt(500)
    mpa = metrics.precisionAt(500)
    map_scores.append(map_)
    ndcg_scores.append(ndcg)
    mpa_scores.append(mpa)
    print(i, 'map score: ', map_, 'ndcg score: ', ndcg, 'mpa score: ', mpa)

output = pd.DataFrame([map_scores, mpa_scores, ndcg_scores])
print('map scores:',map_scores)
print('precision at k:',mpa_scores)
print('ndcg:',ndcg_scores)
output.to_csv('output_ALS.csv')

import time
fit_start_time = time.time()
als = ALS(rank = 40, maxIter=10, regParam=0.1, coldStartStrategy="drop",seed=42)
model = als.fit(train)
fit_end_time = time.time()
total=fit_end_time-fit_start_time
print("Time:",total)
model.save('best_model_for_als')


# Make predictions on test set
train_label = train.select('user', 'item').groupBy('user').agg(expr('collect_list(item) as train_item'))
user_id = test.select('user').distinct()
true_label = test.select('user', 'item').groupBy('user').agg(expr('collect_list(item) as true_item'))
res = model.recommendForUserSubset(user_id,500)
prediction_label = res.select('user','recommendations.item')
condition = [prediction_label.user == train_label.user, prediction_label.item == train_label.train_item]
pred_filtered = prediction_label.join(F.broadcast(train_label), condition, 'left_anti').rdd.map(lambda row: (row[1], row[2]))
pred_true_rdd = prediction_label.join(F.broadcast(true_label), 'user', 'inner').rdd.map(lambda row: (row[1], row[2]))
print('Start Evaluating for test data set')
metrics = RankingMetrics(pred_true_rdd)
map_ = metrics.meanAveragePrecision
ndcg = metrics.ndcgAt(500)


precision_start_time = time.time()
mpa = metrics.precisionAt(500)
precision_end_time = time.time()
print('Precision at 500 time:{}'.format(str(precision_end_time - precision_start_time)))

print('map score: ', map_, 'ndcg score: ', ndcg, 'mpa score: ', mpa)
user_factor = model.userFactors
user_vec = user_factor.toPandas()
user_vec = np.array(list(user_vec['features']))
print("user vector: ",user_vec)
item_factor = model.itemFactors
item_vec = item_factor.toPandas()
item_vec = np.array(list(item_vec['features']))
print("item vector: ",item_vec)
#pkl.dump(item_vec, open('item_vec.pkl','wb'))

