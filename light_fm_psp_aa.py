
import os
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn import preprocessing

os.system("pip install lightfm")
import time
from pyspark.sql import SparkSession
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

def main(spark, train_data_path, val_data_path, test_data_path, loss_functions, num_components, epoch_range):

    train_df = spark.read.parquet(train_data_path)
    train_df.createOrReplaceTempView('train_df')
    validation_df = spark.read.parquet(val_data_path)
    validation_df.createOrReplaceTempView('validation_df')
    test_df = spark.read.parquet(test_data_path)
    test_df.createOrReplaceTempView('test_df')

    # Filtering the validation set such that it contains only books that are already present in the training set
    
    validation_df_filter = spark.sql('SELECT val.user_id, val.book_id, val.rating FROM validation_df val JOIN train_df train on val.book_id = train.book_id ') 
    validation_df_filter.createOrReplaceTempView('validation_df_filter')
    test_df_filter = spark.sql('SELECT test.user_id, test.book_id, test.rating FROM test_df test JOIN train_df train on test.book_id = train.book_id ')
    test_df_filter.createOrReplaceTempView('test_df_filter')

    train_df = train_df.toPandas()
    validation_df = validation_df_filter.toPandas()
    test_df = test_df_filter.toPandas()

    user_id_label_encoded = preprocessing.LabelEncoder()
    user_id_train = user_id_label_encoded.fit_transform(train_df['user_id'].values)
    user_id_val = user_id_label_encoded.transform(validation_df['user_id'].values)
    user_id_test = user_id_label_encoded.transform(test_df['user_id'].values)

    book_id_label_encoded = preprocessing.LabelEncoder()
    book_id_train = book_id_label_encoded.fit_transform(train_df['book_id'].values)
    book_id_val = book_id_label_encoded.transform(validation_df['book_id'].values)
    book_id_test = book_id_label_encoded.transform(test_df['book_id'].values)

    rating_label_encoded = preprocessing.LabelEncoder()
    train_rating = rating_label_encoded.fit_transform(train_df['rating'].values)
    validation_rating = rating_label_encoded.transform(validation_df['rating'].values)
    test_rating = rating_label_encoded.transform(test_df['rating'].values)

    user_latent = len(np.unique(user_id_train))
    item_latent = len(np.unique(book_id_train))
    
    train = coo_matrix((train_rating,(user_id_train, book_id_train)), shape=(user_latent, item_latent))
    validation = coo_matrix((validation_rating,(user_id_val, book_id_val)), shape=(user_latent, item_latent))
    test = coo_matrix((test_rating,(user_id_test,book_id_test)), shape = (user_latent, item_latent))

    output=pd.DataFrame(columns=['loss','components','epochs','time','pak','pak_time','recall','recall_time','auc','auc_time'])
    count=0
    for loss_func in loss_functions:
        for num_comp in num_components:
            for epoch in epoch_range:

                start_time_alg_fit = time.time()
                model = LightFM(no_components=num_comp, loss=loss_func, random_state=22)
                model.fit(train, epochs=epoch) 
                end_time_alg_fit = time.time()

                start_time_precision_calc = time.time()
                precision = precision_at_k(model, validation, k=500).mean()
                end_time_precision_calc = time.time()

                start_time_recall_calc = time.time()
                recall = recall_at_k(model, validation, k=500).mean()
                end_time_recall_calc = time.time()

                start_time_auc_calc = time.time()
                auc = auc_score(model, validation).mean()
                end_time_auc_calc = time.time()

                output.loc[count,:]=[str(loss_func),str(num_comp),str(epoch),str(end_time_alg_fit - start_time_alg_fit),str(precision),str(end_time_precision_calc - start_time_precision_calc), str(recall), str(end_time_recall_calc - start_time_recall_calc),str(auc),str(end_time_auc_calc - start_time_auc_calc)]
                print(output)
                count=count+1


    spark_df = spark.createDataFrame(output)
    spark_df.repartition(1).write.csv('gs://psp334/final_results_5_per.csv')


    # Testing our final results for our best configuration
    test_output=pd.DataFrame(columns=['loss','components','epochs','time','pak','pak_time','recall','recall_time','auc','auc_time'])
    loss_func = 'warp'
    num_comp = 50
    epoch = 20
    start_time_alg_fit = time.time()
    model = LightFM(no_components=num_comp, loss=loss_func, random_state=22)
    model.fit(train, epochs=epoch) 
    end_time_alg_fit = time.time()
    start_time_precision_calc = time.time()
    precision = precision_at_k(model, test, k=500).mean()
    end_time_precision_calc = time.time()
    start_time_recall_calc = time.time()
    recall = recall_at_k(model, test, k=500).mean()
    end_time_recall_calc = time.time()
    start_time_auc_calc = time.time()
    auc = auc_score(model, test).mean()
    end_time_auc_calc = time.time()
    test_output.loc[0,:]=[str(loss_func),str(num_comp),str(epoch),str(end_time_alg_fit - start_time_alg_fit),str(precision),str(end_time_precision_calc - start_time_precision_calc), str(recall), str(end_time_recall_calc - start_time_recall_calc),str(auc),str(end_time_auc_calc - start_time_auc_calc)]
    
    spark_df = spark.createDataFrame(test_output)
    spark_df.repartition(1).write.csv('gs://psp334/final_test_results_5_per.csv')


if __name__ == "__main__":
    spark = SparkSession.builder.appName('lightfm').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    train_data_path = 'gs://psp334/Train_data_five.parquet/part-00000-56c0e310-3b43-4001-9c21-ea46c0f26c81-c000.snappy.parquet'
    val_data_path = 'gs://psp334/Val_data_five.parquet/part-00000-87162e7e-a3d3-4e88-bf3d-fa897675136a-c000.snappy.parquet'
    test_data_path = 'gs://psp334/Test_data_five.parquet/part-00000-08307416-dae3-49d3-a3d4-07571c163978-c000.snappy.parquet'
    loss_functions = ['warp','bpr', 'logistic']
    num_components = [10,20,30,40,50]
    epoch_range = [5,10,15,20,25,30,35]
    main(spark, train_data_path, val_data_path, test_data_path, loss_functions, num_components, epoch_range)
    spark.stop()