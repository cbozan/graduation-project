from pyspark.sql import SQLContext # al
from pyspark import SparkContext # al
from pyspark.ml import Pipeline# al
from pyspark.ml.classification import RandomForestClassifier# al
from pyspark.ml.feature import HashingTF, Tokenizer# al
import pandas as pd # al


from pyspark.sql.functions import col

# mllib
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from pyspark.ml.classification import LogisticRegression

# mllib, for pipeline
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

# import schedule


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#id_to_category = pd.read_csv("data/id_to_category.csv", index_col=[0]).to_dict()['0']

id_to_category = pd.read_csv('data/category_to_id.csv', index_col=1, header=None).squeeze(1).to_dict()


print("SparkContext hazırlanıyor...") #
sc = SparkContext() # al

print("SQLContext hazırlanıyor...")
sqlContext = SQLContext(sc) # al

data = sqlContext.read.format('com.databricks.spark.csv').options(header='true',
    inferschema='true').load("split-data/file-for-spark-0.csv") # al

# istenmeyen sütunları kaldır.
data = data.select([column for column in data.columns if column != "_c0"]) # al

# category sütunu başa getirildi (text, category) -> (category -> text)
cols = data.columns # al
cols = cols[-1:] + cols[:-1] # al
data = data[cols] # al


tokenizer = Tokenizer(inputCol="text", outputCol="words") # al
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features") # al
rf = RandomForestClassifier(labelCol="category", featuresCol="features", numTrees=10) # al
pipeline = Pipeline(stages=[tokenizer, hashingTF, rf]) # al

model = pipeline.fit(data)


""
# Kategorilere göre gruplanması ve azalan sırada yazdırılması
#print("Kategorilere göre gruplanması ve azalan sırada yazdırılması")
#print( data.groupBy("category").count().orderBy(col("count").desc()).show())

# Verilerin category_id'ye göre gruplanması ve ilk 10 tane verinin yazdırılması
#data.groupBy('category_id').count().orderBy(col("count").desc()).show(10)


# Veri temizleme (RegexTokenizer, StopWordsRemover, CountVectorizer)
# bu aşamaların tümü pandas kütüphaneleriyle yapıldı.



"""
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")

tr_stopwords = []
stopWordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered").setStopWords(tr_stopwords)
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=10000, minDF=5)
labelCategory = StringIndexer(inputCol="category", outputCol="label")

pipeline = Pipeline(stages=[regexTokenizer, stopWordsRemover, countVectors, labelCategory])

pipelineFit = pipeline.fit(data)
dataset = pipelineFit.transform(data)
print("Veri seti ilk 5 değeri\n", dataset.show(5))


# Eğtim ve test verileri - seed: skileardeki random_state karşılığı
trainData, testData = dataset.randomSplit([0.7, 0.3], seed=142)
print("Eğitim veri sayısı: ", trainData.count())
print("Test veri sayısı: ", testData.count())

# model değerlendirmeleri

#>>> lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
#>>> lrModel = lr.fit(trainingData)
#predictions = lrModel.transform(testData)
#>>> predictions.filter(predictions['prediction'] == 0).select("text", "category", "probability", "label", "pre
#diction").orderBy("probability", ascending=False).show(n=10, truncate=30)
#
#
#>>> evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
#>>> evaluator.evaluate(predictions)


lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
lrModel = lr.fit(trainData)
predictions = lrModel.transform(testData)
predictions.filter(predictions['prediction'] == 0).select("text", "category_id", "probability", "prediction").orderBy("probability", ascending=False).show(5)


from pyspark.sql import SparkSession, Row
spark = SparkSession.builder.config("spark.driver.memory", "15g").getOrCreate()
#rdd = spark.sparkContext.parallelize([trainData.collect()[0]])
tmp = spark.createDataFrame([trainData.collect()[0]])
"""
"""
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import HashingTF, Tokenizer
regexTokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
hashingTF = HashingTF(inputCol=regexTokenizer.getOutputCol(), outputCol="features")
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10)
labelCategory = StringIndexer(inputCol="category", outputCol="label")
pipeline = Pipeline(stages=[regexTokenizer, hashingTF, labelCategory, rf])
model = pipeline.fit(data)
"""


"""program_control = True

id_to_category = pd.read_csv("data/id_to_category.csv", index_col=[0])
i = 0


def user_schedule():
    program_control = (input("Servis çalışıyor sonlandırmak için -1 girin: ") != "-1")
    user_schedule()

def realtime_data():
    data = pd.read_csv("split-data/file-for-spark-{}.csv".format(i), index_col=[0])
    if i < 19:
        i += 1
    process_data(data)

def process_data(data):
    print(data)

scheduler1 = schedule.Scheduler()
scheduler1.every(0.2).minutes.do(realtime_data)



while program_control:
    scheduler1.run_pending()
else:
    exit()"""
