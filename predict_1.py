import pickle
from flask import Flask
from flask import request
from flask import jsonify
from sklearn import model_selection
import pandas

model_file = 'finalized_model.sav'
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

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)


app = Flask('ping')

@app.route('/predict', methods=['POST'])
def predict(X_test):
    
    x_test = request.get_json()

    
    Y_pred = model.predict(X_test)
    return Y_pred





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

