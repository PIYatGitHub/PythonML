from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
# import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.datasets import fetch_20newsgroups
# #
# # from nltk.corpus import gutenberg
# # macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
# # # print(len(macbeth_sentences))
# # print(macbeth_sentences.data)
# #
# #
# # # cv = CountVectorizer(stop_words="english", max_features=500)
# # # groups = fetch_20newsgroups()
# # # transformed = cv.fit_transform(groups.data)
# # # print(cv.get_feature_names())
# # #
# # # sns.distplot(np.log(transformed.toarray().sum(axis=0)))
# # # plt.xlabel('Log Count')
# # # plt.ylabel('Frequency')
# # # plt.title('Distribution Plot of 500 Word Counts')
# # # plt.show()


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import gutenberg

macbeth_words = gutenberg.words('shakespeare-macbeth.txt')
print(len(macbeth_words))
cv = CountVectorizer(stop_words="english", max_features=500)
# groups = fetch_20newsgroups()
transformed = cv.fit_transform(macbeth_words)
print(cv.get_feature_names())
#
sns.distplot(np.log(transformed.toarray().sum(axis=0)))
plt.xlabel('Log Count')
plt.ylabel('Frequency')
plt.title('Distribution Plot of 500 Word Counts')
plt.show()
