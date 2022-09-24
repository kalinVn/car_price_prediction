from car_price_prediction_custom.visualizator import plot_actual_predicted_prices
from car_price_prediction_custom.App import App as AppCustom

CSV_FILE_PATH = 'datasets/car_data.csv'


def car_prediction_custom():
    app = AppCustom(CSV_FILE_PATH)
    app.standardize_data()
    app.fit()
    app.predict()
    app.error_score()

    y_test = app.get_test_prediction_lin_model()
    test_prediction_lin_model = app.get_test_prediction_lin_model()
    plot_actual_predicted_prices(y_test, test_prediction_lin_model)


car_prediction_custom()




