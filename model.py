import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
# print(X)
# print(len(Y))

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model
model = LogisticRegression()
model.fit(X_train, Y_train)

# save model
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
# with open('model.bin', 'wb') as f_out:
#     pickle.dump((dict_vectorizer, model), f_out)
# f_out.close()


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
predict = loaded_model.predict(X_test)

result = loaded_model.score(X_test, Y_test)

# print(result)
# with open('mode.bin', 'rb') as f_in:
#     dict_vectorizer, model = pickle.load(f_in)
# f_in.close()
