import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv(
  filepath_or_buffer='training.1600000.processed.noemoticon.csv',
  encoding="ISO-8859-1",
  usecols=[0, 5],
  names=['target', 'text'],
  engine='python',
)

df['text'] = df['text'] \
	.str.replace(r'(?:@|#|https?:|www\.)\S+', '') \
	.str.replace(r'[^A-Za-z0-9 ]+', '') \
	.str.split() \
	.str.join(' ') \
	.str.lower()

print('preprocessing done')

df['target'] = df['target'].replace(4, 1)




X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'],
                                                      test_size=0.1,
                                                      random_state=123,
                                                      stratify=df['target'])

NGRAM_RANGE = 2
MAX_ITER = 1000

vectorizer = CountVectorizer(ngram_range=(1, NGRAM_RANGE))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

log_reg = LogisticRegression(max_iter=MAX_ITER, random_state=123)
log_reg = log_reg.fit(X_train, y_train)

print('model fitted')

file = open('vectoriser-ngram-(1,2).pickle','wb')
pickle.dump(vectorizer, file)
file.close()

file = open('Sentiment-LR.pickle','wb')
pickle.dump(log_reg, file)
file.close()

print('done with model')


