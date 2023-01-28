# CRISP-DM VERİNİN HAZIRLANMASI aşaması
import pandas as pd
import numpy as np


rawData = pd.read_csv('data/data-set.csv')

rawData['category'] = rawData['category'].astype('category')
#rawData['category'] = rawData['category'].cat.codes
# Tekrar eden kayıtların silinmesi
print("Tekrar eden kayıtların silinmesi")
print("-" * 50)
print(rawData.text.duplicated(keep="first").value_counts())
rawData.drop_duplicates(subset="text", keep="first", inplace=True, ignore_index=True)
print("\n")

# Yeni Satır Sayısı
print("\nSatır sayısı")
print("-" * 50)
print(len(rawData.index))
print("\n")

# Text sütununda bulunan kelime sayısını gösteren kelime adlı bir sütun ekle
rawData.insert(2, "kelime", [len(data.split()) for data in rawData['text'].tolist()],True)

# Uç değerlerin analizi
desc = rawData['kelime'].describe()
minHesap = round(desc['mean'] - desc['std'])
maxHesap = round(desc['mean'] + desc['std'])
for i in range(minHesap, maxHesap + 1):
    print("{} - {} arasinda {} kayıt".format(i, maxHesap+1, len(rawData[rawData['kelime']>i]) - len(rawData[rawData['kelime']>maxHesap+1])))

# Analiz sonucunda max min uygun kabul edildi
rawData = rawData[rawData['kelime'] > minHesap]
rawData = rawData[rawData['kelime'] < maxHesap]

# Kelime istatistiklerini dışa aktar (sunum için)
desc = rawData.kelime.describe()
desc = desc.drop('count').astype(np.int64)
desc.to_csv('data/kelime-istatistikleri2.csv')

# Yeni veri seti
print("\nYeni veri seti bilgileri")
print("-" * 50)
print(rawData.describe())
print("\n")

# türkçe karakter, büyük küçük harf ve gereksiz kelimelerin çıkarılması işlemleri
trEng = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
rawData["text"] = rawData["text"].str.translate(trEng)
rawData["text"] = rawData["text"].str.lower()
rawData["text"] = rawData["text"].str.replace("[^a-zA-Z\s]", " ", regex=True) # re.sub(r"[^a-zA-Z\s]", " ", rawData.text[0])
rawData["text"] = rawData["text"].str.replace("devamini oku", "", regex=False) # temp = re.sub("devamini oku", "", temp)
rawData["text"] = rawData["text"].str.replace("[ \t]{2,}", " ", regex=True) # re.sub(r"(\s\s)+", " ", bioncekitext)

# stopwords kelimelerini veri seti içerisinden temizlenmesi
stopwords = pd.Series([satir.strip() for satir in open("data/stopwords.txt", mode='r', encoding='utf-8')])
stopwords = stopwords.str.translate(trEng)
stopwords = pd.Series(list(set(stopwords)))
rawData['text'] = [' '.join([l for l in x.split() if l not in stopwords]) for x in rawData['text']]

# kelime analizi
kelimeFrekans = pd.Series(np.concatenate([x.split() for x in  rawData['text']])).value_counts()

desc = kelimeFrekans.describe()
desc = desc.drop('count').astype(np.int64)
desc.to_csv('data/kelime-frekansi1.csv')

# kelimelerin frekansını hesapla
ustKelimeFrekans = kelimeFrekans[kelimeFrekans > 7]
altKelimeFrekans = kelimeFrekans[kelimeFrekans <= 7]

# ust ve alt kelimeleri dışa aktar
ustKelimeFrekans.to_csv('data/ustkelime-data.csv')
altKelimeFrekans.to_csv('data/altkelime-data.csv')

# çok fazla tekrarlayan kelimeler silindikten sonra frekans istatistikleri (sunum için)
desc = altKelimeFrekans.describe()
desc = desc.drop('count').astype(np.int64)
desc.to_csv('data/kelime-frekansi2.csv')

desc = ustKelimeFrekans.describe()
desc = desc.drop('count').astype(np.int64)
desc.to_csv('data/kelime-frekansi3.csv')

# Veriyi karıştır ve dışarı aktar
rawData = rawData.sample(frac=1)
rawData.to_csv("data/data-set2.csv", index=False)
