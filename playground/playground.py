import numpy
from sklearn.naive_bayes import GaussianNB
from pandas import *
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dictionary = {'name': Series(['Braud', 'Cummings', 'Heikken', 'Allen'], index=['a', 'b',
'c', 'd']), 'age': Series([22,38,26,35], index=['a', 'b', 'c', 'd']),
     'fare': Series([7.25, 71.83, 8.05], index=['a', 'b','d']),
     'survived?': Series([False, True, True, False], index=['a', 'b','c', 'd'])}

dataFrame = DataFrame(dictionary)

countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea',
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

sochiOlympics = {'countryName': Series(countries), 'gold': Series(gold),
                 'silver': Series(silver), 'bronze': Series(bronze)}

olympic_medal_counts_df = DataFrame(sochiOlympics)

at_least_1_gold = olympic_medal_counts_df[ olympic_medal_counts_df['gold'] >= 1]
avg_bronze_with_1_gold = numpy.mean(at_least_1_gold['bronze'])

mean_all_columns = olympic_medal_counts_df[['gold', 'silver', 'bronze']].apply(numpy.mean)

scores_by_medals = [4, 2, 1]
medal_counts = olympic_medal_counts_df[['gold', 'silver', 'bronze']]
points = numpy.dot(medal_counts, scores_by_medals)

sochiOlympicsScores = {'countryName': Series(countries), 'score': Series(points)}

sochiOlympicsScores_df = DataFrame(sochiOlympicsScores)


X = numpy.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = numpy.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB()
clf.fit(X, Y)
GaussianNB(priors=None, var_smoothing=1e-09)
print(clf.predict([[-0.8, -1]]))


iris = datasets.load_iris()
features = iris.data
labels = iris.target
features_train, features_test, labels_train, labels_test = train_test_split(
iris.data, iris.target, test_size=0.4, random_state=0)
clf = SVC(kernel="linear", C=1.)
clf.fit(features_train, labels_train)
print (clf.score(features_test, labels_test))

def submitAcc():
    return clf.score(features_test, labels_test)