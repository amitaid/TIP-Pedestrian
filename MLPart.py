# Load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
path = "C:\\ImageProcessingProject\\lightweight-human-pose-estimation-3d-demo.pytorch-master\\3d\\Poses\\posesData.csv"     # url or just path
dataset = read_csv(path)
print(dataset.shape)                                 # this information will help in lines 22-23

# Split-out validation dataset
array = dataset.values
X = array[:, 0:57]
y = array[:, 58]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Make predictions on validation dataset
model = KNeighborsClassifier()                              # using now the KNN model
model.fit(X_train, Y_train)                                 # fitting (learning) the training set
predictions = model.predict(X_validation)                   # and lastly predicting the labels for the v-set
# Evaluate predictions
print(accuracy_score(Y_validation, predictions))
