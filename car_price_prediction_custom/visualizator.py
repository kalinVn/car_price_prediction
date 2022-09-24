
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def plot_actual_predicted_prices(actual_prices, predicted_prices):
    plt.scatter(actual_prices, predicted_prices)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual Prices vs Predicted Prices")
    plt.show()


