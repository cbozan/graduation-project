# CRISP-DM VERİNİN ANLAŞILMASI aşaması
import pandas as pd
import csv
import numpy as np

rawData = pd.read_csv('data/data-set.csv')

# Satır ve sütün sayısı
print("\nSatır ve sütün sayısı")
print("-" * 50)
print(rawData.shape)
print("\n")

# Eksik veri kontrolü gerçekleştirilir.
print("Eksik veri kontrolü:")
print("-" * 50)
print(rawData.info())
print("\n")

# Satırlar benzersiz mi?
print("Satırlar benzersiz mi?")
print("-" * 50)
print(rawData.text.duplicated(keep="first").value_counts())
print("\n")

# Categorilerin incelenmesi
print("Kategorilerin incelenmesi")
print("-" * 50)
print("Sayısı : ", len(rawData.category.unique()))
print("\n")
print("Her bir kategoriye ait kayıt sayısı")
print("-" * 40)
print(rawData.category.value_counts())
print("\n")


# Kategorilerin içerdiği veri sayısını csv olarak dışa aktarılması (sunum için)
categoryAnaliz = rawData.category.value_counts()
categoryAnaliz.to_csv(path_or_buf="data/category-count.csv")

category_istatistik = (categoryAnaliz.describe() / 100).astype(np.int64)
category_istatistik = category_istatistik.drop('count')
category_istatistik.to_csv("data/category-istatistik.csv")

# Her satırın kelime sayısının incelenmesi
print("Her satırın kelime sayısının incelenmesi")
print("-" * 50)
rawData.insert(2, "kelime", [len(data.split()) for data in rawData['text'].tolist()],True)
print(rawData.head(4))
print("\n")
print("Kelime sayısı analizi")
print("-" * 50)
print(rawData['kelime'].describe())
print("\n")

# Verilerin istatiski bilgilerinı csv olarak dışa aktarılması (sunum için)
desc = rawData.kelime.describe()
desc = desc.drop('count').astype(np.int64)
desc.to_csv("data/kelime-istatistikleri.csv")

desc = rawData.kelime.value_counts()
desc = desc.drop
