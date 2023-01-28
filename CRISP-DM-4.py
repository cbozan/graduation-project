import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

data = pd.read_csv("data/data-set2.csv")
print("*"*10, "VERI SETI", "*"*10)
print(data)
print("\n")

# categoryleri birer sayı ile temsil edecek olan sütunu ekle
data['category'] = data['category'].astype('category')
data['category_id'] = data['category'].cat.codes

# kategoriden id'ye id'den kategoriye hızlı erişim için sözlükler oluşturma
id_to_category = pd.Series(data.category.values, index=data.category_id).to_dict()
category_to_id = {v:k for k,v in id_to_category.items()}

features, targets = data['text'], data['category_id']

#           Verileri Eğitim ve Test olarak ayırma
# train_size -> eğitime ayrılacak veri oranı (max 1)
# test_size -> teste ayrılaca veri oranı (1 - train_size)
# random_state -> rastgeleleği isimlendirmek (bir numara atamak)
# shuffle -> verileri karıştırmak
# stratify -> verilerin targets değerlerine göre eşit dağılımını sağlamak
all_train_features, test_features, all_train_targets, test_targets = train_test_split(
    features, targets,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=targets
)

# Verilerin kırpılma oranı
reduce_ratio = 0.05
reduced_train_features, _, reduced_train_targets, _= train_test_split(
    all_train_features, all_train_targets,
    train_size=reduce_ratio,
    random_state=42,
    shuffle=True,
    stratify=all_train_targets
)

reduced_test_features, _, reduced_test_targets, _ = train_test_split(
    test_features, test_targets,
    train_size=reduce_ratio,
    random_state=42,
    shuffle=True,
    stratify=test_targets
)

train_features, val_features, train_targets, val_targets = train_test_split(
    reduced_train_features, reduced_train_targets,
    train_size=0.9,
    random_state=42,
    shuffle = True,
    stratify=reduced_train_targets
)

reducedDataAnalizIndex = [
    "Train", "Test", "Validation"
    ]

reducedDataAnaliz = [
    len(train_features), len(reduced_test_targets), len(val_targets)
    ]

reducedDataAnalizSeries = pd.Series(reducedDataAnaliz, index=reducedDataAnalizIndex)
reducedDataAnalizSeries.to_csv("data/azaltilmis-veri-analizi.csv")

#           Vektörleştirme
# sublinear_tf -> 1+log() ölçeklendirmesi
# min_df -> bu sayıdan küçük olan terimleri yoksay
# nomr -> normalizasyon , l2 -> Bir verideki (satırdaki) her bir sayının karesinin
#       toplamı 1 olacak şekilde ölçeklendiren bir normalizasyon türü
# ngram_range -> unigram ve bigram ' ları temsil eder.
#       unigram -> tek kelimeleri ifade eder
#       bigram -> ikili kelimeleri ifade eder
#       (1, 1) -> sadece unigram, (2, 2) -> sadece bigram, (1, 2) -> unigram ve bigram
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2))
train_features_transform = tfidf.fit_transform(train_features).toarray()
labels = train_targets

# chi2 -> ki-kare testi (chi-square test)
N = 2
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(train_features_transform, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = tfidf.get_feature_names_out()[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(category))
  print("  İlişkili ilk {} unigram:\n. {}".format(N, '\n. '.join(unigrams[:N])))
  print("  İlişkili son {} unigram:\n. {}".format(N, '\n. '.join(unigrams[-N:])))
  print("  İlişkili ilk {} bigram:\n. {}".format(N, '\n. '.join(bigrams[:N])))
  print("  İlişkili son {} bigram:\n. {}".format(N, '\n. '.join(bigrams[-N:])))
  print("")


# Verileri Naive Bayes Sınıflandırması ile eğitmek
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
train_feature_counts = count_vect.fit_transform(train_features)
train_features_tfidf = tfidf_transformer.fit_transform(train_feature_counts)
clf = MultinomialNB().fit(train_features_tfidf, train_targets)



models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, train_features_transform, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

#import seaborn as sns
#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#              size=8, jitter=True, edgecolor="gray", linewidth=2)
#plt.show()
cv_df = cv_df.groupby('model_name').accuracy.mean()
cv_df.to_csv("data/model-dogruluk-oranlari.csv")

model = LinearSVC()
vectorizer = CountVectorizer()
tfidf = TfidfTransformer()
train_features_vect = vectorizer.fit_transform(train_features)
train_features_tfidf = c_tfidf.fit_transform(train_features_vect)
model = model.fit(train_features_tfidf, train_targets)


test_targets_vect = vectorizer.transform(reduced_test_features)

test_result = model.predict(test_targets_vect)
test_result = pd.Series(test_result)
real_result = reduced_test_targets

test_result_group_by_category = test_result.value_counts().sort_index()
real_result_group_by_category = real_result.value_counts().sort_index()

test_result_group_by_category.to_csv("data/test_result_group_by_category.csv")
real_result_group_by_category.to_csv("data/real_result_group_by_category.csv")

metr=metrics.classification_report(test_result, real_result, target_names=data['category'].unique(), output_dict=True)
metr = pd.DataFrame(metr).transpose()
metr.to_csv("data/test_result_metrics.csv")


# testGiris = input("Kategorisi belirlenecek bir veri girin:")
# print("Metin {} kategoriye ait".format(id_to_category[model.predict(c_vect.transform([text]))[0]]))

# d_targets = reduced_test_targets.copy()


""""
 def pred(text):
...     count = count_vect.fit_transform(text)
...     tf = tfidf_transformer.fit_transform(count)
...     print("sonucc : ", model.predict(tf))
...
>>> def run():
...     i = input("Text ya da null (cikis): ")
...     if i == None:
...             exit()
...     else:
...             pred(i)
"""""
