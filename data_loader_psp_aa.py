import sys

from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from operator import itemgetter
from itertools import groupby
import pandas as pd
from pyspark.sql.types import *
import itertools
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import OneHotEncoder, StringIndexer
import numpy as np




memory = '32g' 
spark = (SparkSession.builder
         .appName('train_als')
         .master('yarn')
         .config('spark.executor.memory', memory)
         .config('spark.driver.memory', memory)
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

data = spark.read.csv('goodreads_interactions.csv',header = True)
data.createOrReplaceTempView('data')
data = spark.sql('select user_id, rating, book_id from data where int(user_id)%20 = 0')
data = data[data['rating']!=0]
user_df = data.groupby('user_id')
a = user_df.count()
data.createOrReplaceTempView('data')

b = a[a['count']>20]
b.createOrReplaceTempView('b')

book_df = data.groupby('book_id')
c = book_df.count()
d = c[c['count']>15]
d.createOrReplaceTempView('d')

inner_join = spark.sql('Select b.user_id, data.rating, d.book_id from data, b, d where data.user_id = b.user_id and data.book_id = d.book_id')
check=inner_join.groupby('user_id').count()
check.createOrReplaceTempView('check')
inner_join.createOrReplaceTempView('inner_join')
(training, tune,  test) = check.randomSplit([0.6, 0.2, 0.2], seed = 0)
training.createOrReplaceTempView('training')
tune.createOrReplaceTempView('tune')
test.createOrReplaceTempView('test')
train_set = spark.sql('Select training.user_id, inner_join.rating, inner_join.book_id from inner_join, training where inner_join.user_id = training.user_id')
val_set = spark.sql('Select tune.user_id, inner_join.rating, inner_join.book_id from inner_join, tune where inner_join.user_id = tune.user_id')
test_set = spark.sql('Select test.user_id, inner_join.rating, inner_join.book_id from inner_join, test where inner_join.user_id = test.user_id')
from pyspark.sql.window import Window
window = Window.partitionBy('user_id').orderBy('book_id')
import pyspark.sql.functions as F
val =(val_set.select("user_id","book_id","rating", F.row_number().over(window).alias("row_number")))
test =(test_set.select("user_id","book_id","rating", F.row_number().over(window).alias("row_number")))
val.createOrReplaceTempView('val')
test.createOrReplaceTempView('test')
val_data = spark.sql('Select * from val where row_number%2 = 0')
test_final = spark.sql('Select * from test where row_number%2 = 0')
val_train = spark.sql('Select * from val where row_number%2 = 1')
test_train = spark.sql('Select * from test where row_number%2 = 1')
val_train = val_train.drop('row_number')
val_data = val_data.drop('row_number')
test_final = test_final.drop('row_number')
train_set = train_set.unionByName(val_train)
test_train = test_train.drop('row_number')
train_data = train_set.unionByName(test_train)
print('data_generated')
train_data.repartition(1).write.parquet('Train_data_two.parquet')
val_data.repartition(1).write.parquet('Val_data_two.parquet')
test_final.repartition(1).write.parquet('Test_data_two.parquet')
