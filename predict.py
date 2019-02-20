import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

train_comments = train['comment'].fillna('none').values
y_train = train['label'].values

test_comments = test['comment'].fillna('none').values

tfidf = TfidfVectorizer(analyzer='word',min_df=2, max_df=0.8, max_features=10000, norm='l2')

X_train = tfidf.fit_transform(train_comments)
X_test = tfidf.transform(test_comments)

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=69)
fold_x = folds.split(X_train, y_train)
test_id = test['id'].values
model = SVC(kernel='rgf', gamma=.08)
model.fit(X_train, y_train)
scores = cross_validate(model, X_train, y_train, cv=fold_x)
y_test = model.predict(X_test)

submission_df = {"id": test_id,
                 "label": y_test}
submission = pd.DataFrame(submission_df)
submission.to_csv('subission.csv')
X1_train, X1_test, y1_train, y1_test = train_test_split(X_train, y_train, test_size= 13000)
y_pred = model.predict(X1_test)
print("Accuracy: %.2f %%" %(100*accuracy_score(y1_test, y_pred)))



