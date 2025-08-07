# Car-Price-Prediction-App
This project is a machine learning-powered web application built with Streamlit and Random Forest Regressor. It predicts the price of a car based on specifications like manufacturer, production year, engine volume, mileage, fuel type, and number of cylinders. It also features interactive data visualizations to explore pricing trends and car market insights.

##  Features
- Predict used car prices using machine learning

- User-friendly interface with Streamlit

- Automatic handling of categorical data via One-Hot Encoding

- Real-time visual analytics powered by Matplotlib and Seaborn

- Three key interactive charts:

  Top Manufacturers by listing count (Bar chart)

  Average Car Price by Production Year (Line chart)

  Mileage vs. Price relationship (Scatter plot)

- Saves trained model to disk using Pickle for fast reloading

---

## Technologies Used
- Python
The core programming language used to build the app.

Handles everything: data processing, ML training, UI logic, and visualization.

- Pandas
Used for data loading, cleaning, filtering, and transformation.

Converts string values to usable numerical formats (Engine volume, Mileage).

Drops irrelevant features like ID, Levy, and Model.

- Scikit-learn
Provides the RandomForestRegressor used to train the prediction model.

Handles regression logic and is robust to overfitting due to ensemble learning.

- Streamlit
Used to build the web app's user interface.

Provides components like selectbox, number_input, and layout utilities for interactivity.

Displays prediction results and visual charts.

- Matplotlib & Seaborn
Used together to plot clean, customizable charts:

Bar charts for top manufacturers

Line charts for average price per year

Scatter plots for price vs. mileage

- Pickle
Saves the trained model (rf_model.pkl) and the model’s feature columns (model_columns.pkl) for reuse.

Helps avoid retraining the model every time the app runs.

## Code & Datasets
- [train model.py](https://github.com/A-lehlouh/Car-Price-Prediction-App/blob/main/price.py)
- [datasets](https://github.com/A-lehlouh/Car-Price-Prediction-App/blob/main/car_price_prediction%20(Autosaved).csv)
-[app.py]()

---

## Contact
If you have any questions, feel free to reach out via GitHub Issues or email: [AbdulrahmanLehlouh@gmail.com]

## License
This project is open-source and available under the MIT License.
