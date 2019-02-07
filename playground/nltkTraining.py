# prepare the nltk and download its large datasets
import nltk
nltk.download()
# import the names corpus
from nltk.corpus import names
print (names.words()[:10])
print (len (names.words()))

# import the PorterStemmer algorithm from the appropriate dir
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
porter_stemmer.stem('machines')
# outputs 'machin'
porter_stemmer.stem('learning')
# outputs 'learn'

# import a lemamtization algorithm
from nltk.stem import WordNetLemmatizer
lemamtizer = WordNetLemmatizer()
lemamtizer.lemmatize('machines')
# outputs 'machine'
lemamtizer.lemmatize('learning')
# outputs 'learning', because it acts only on NOUNS!!!
