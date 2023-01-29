from pyspark.sql import SparkSession, Row

import pyspark.ml.feature
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StringType

#from pyspark.ml.classification import RandomForestClassifier
#from pyspark.ml.feature import HashingTF, Tokenizer
import pandas as pd
import sys

def sparkSession():
    print("Spark oturumu başlatılıyor...\n")
    print("spark baslatiliyor.")
    spark = SparkSession.builder.appName("BitirmeProjesi").getOrCreate()
    print("heyyy")
    return spark

def readData(spark, file_path):
    print("*" * 50, '"{}" dosyası yükleniyor...'.format(file_path), "*" * 50, sep="\n", end="")
    data = spark.read.format('com.databricks.spark.csv')\
        .options(header='true', inferschema='true')\
        .load(file_path)
    data = data.select('category', 'text')

    print("*" * 50, 'Kodlanmış etiket dönüşümleri için sözlük oluşturuluyor...', "*" * 50, sep="\n", end="")

    labelEncoder = StringIndexer(inputCol='category', outputCol='label').fit(data)
    data = labelEncoder.transform(data)

    id_to_category_df = data.groupBy('category', 'label').count().select('category', 'label').toPandas()
    id_to_category_dict = {}
    for i in range(len(id_to_category_df)):
        id_to_category_dict[id_to_category_df.label[i]] = id_to_category_df.category[i]

    return data, id_to_category_dict

def createPipeline():
    print("*" * 50, "Pipeline oluşturuluyor...", "*" * 50, sep="\n", end="")
    tokenizer = Tokenizer(inputCol = 'text', outputCol='words')

    stop_words_file = open("data/stopwords.txt", "r")
    stop_words = stop_words_file.read()
    stopwordList = stop_words.split("\n")
    stopwords_remover = StopWordsRemover(inputCol='words', outputCol='filtered', stopWords=stopwordList)

    vectorizer = CountVectorizer(inputCol='filtered', outputCol='featuresVectorizer')
    idf = IDF(inputCol='featuresVectorizer', outputCol='features')
    lr = LogisticRegression(featuresCol='features', labelCol='label')

    return Pipeline(stages=[tokenizer, stopwords_remover, vectorizer, idf, lr])

def sess(trend):
    spark = sparkSession()
    data, id_to_category = readData(spark, 'split-data/file-for-spark-0.csv')
    pipeline = createPipeline()


    print("*" * 50, "Veri eğitiliyor, Model oluşturuluyor...", "*" * 50, sep="\n", end="")
    train, test = data.randomSplit((0.7, 0.3), seed=42)
    model = pipeline.fit(train)
    predictions = model.transform(test)

    print("*" * 50, "*" * 50, "Model oluşturuldu", "*" * 50, "*" * 50, sep="\n", end="")
    evaluator = MulticlassClassificationEvaluator(labelCol='label',predictionCol='prediction',metricName='accuracy')
    accuracy = evaluator.evaluate(predictions)
    print("-" * 50, "-" * 50, "Model başarımı: {}".format(accuracy), "-" * 50, "-" * 50, sep="\n", end="")

    trend[''] = StringType()
    data = spark.createDataFrame(trend)
    predictions = model.transform(data)
    predictions = predictions.select('text', 'prediction')

    predictions = predictions.toPandas()
    predictions['prediction'] = pd.Series([id_to_category[x] for x in predictions['prediction']])

    return predictions
