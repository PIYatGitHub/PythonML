from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
macbeth_words = gutenberg.words('shakespeare-macbeth.txt')

def letters_only(astr):
    return astr.isalpha()

cv = CountVectorizer(stop_words="english", max_features=500)
#groups = fetch_20newsgroups()
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in macbeth_words:
    cleaned.append(' '.join([lemmatizer.lemmatize(word.lower())
                             for word in post.split()
                             if letters_only(word)
                             and word not in all_names]))

transformed = cv.fit_transform(cleaned)
km = KMeans(n_clusters=1000)
km.fit(transformed)
labels = cleaned
print('labels', len(labels))
print('km.labels_', len(km.labels_))
plt.scatter(labels, km.labels_)
plt.xlabel('Newsgroup')
plt.ylabel('Cluster')
plt.show()
