
import pandas as pd
from sklearn.model_selection import train_test_split
import random as rnd
import os

print("data-set2.csv dosyası yükleniyor..")
data = pd.read_csv("data/data-set2.csv")

text, category = data['text'], data['category']

split_number = 50
reduced_ratio = 1 / split_number
print("Verileri ayrıştırma işlemi başlıyor")
if not os.path.exists('./split-data'):
    os.mkdir('./split-data')

for i in range(0, split_number):
    s_text, _, s_category, _ = train_test_split(text, category, train_size=reduced_ratio, random_state=rnd.randint(0, 234234), shuffle=True, stratify=category)
    split_file = pd.DataFrame([s_text, s_category]).transpose()
    split_file.to_csv("split-data/file-for-spark-{}.csv".format(i))
    print("split-data/file-for-spark-{}.csv dosyası oluşturuldu.".format(i))

print("\nİşlem tamamlandı.")
