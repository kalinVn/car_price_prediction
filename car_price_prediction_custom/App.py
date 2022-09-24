import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


class App:

    def __init__(self, csv_path):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.csv_path = csv_path
        self.car_dataset = pd.read_csv(csv_path)

        self.lin_reg_model = LinearRegression()
        self.lasso_model = Lasso()

        self.training_prediction_lin_model = None
        self.training_prediction_lasso_model = None
        self.test_prediction_lin_model = None
        self.test_prediction_lasso_model = None

    def standardize_data(self):
        self.car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
        self.car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
        self.car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

        self.x = self.car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
        self.y = self.car_dataset['Selling_Price']

    def fit(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.1,
                                                                                random_state=2)

        self.lin_reg_model.fit(self.x_train, self.y_train)
        self.lasso_model.fit(self.x_train, self.y_train)

    def predict(self):
        self.training_prediction_lin_model = self.lin_reg_model.predict(self.x_train)
        self.test_prediction_lin_model = self.lin_reg_model.predict(self.x_test)
        self.training_prediction_lasso_model = self.lasso_model.predict(self.x_train)
        self.test_prediction_lasso_model = self.lasso_model.predict(self.x_test)

    def error_score(self):
        self.show_error_score(self.y_train, self.training_prediction_lin_model)
        self.show_error_score(self.y_test, self.test_prediction_lin_model)
        self.show_error_score(self.y_train, self.training_prediction_lasso_model)
        self.show_error_score(self.y_test, self.test_prediction_lasso_model)

    def show_error_score(self, y, data_prediction):
        error_score = metrics.r2_score(y, data_prediction)
        print("R scored Error: ", error_score)

    def get_y_test(self):
        return self.y_test

    def get_test_prediction_lin_model(self):
        return self.test_prediction_lin_model

