# Car-Price-Prediction-App
 This is a Streamlit-based machine learning web app that predicts the market price of a used car based on key input features like manufacturer, engine volume, fuel type, mileage, and production year. The project also includes visual analytics to explore market trends and gain insights into car pricing patterns.

##  Features

-  Predict used car prices using machine learning
-  User-friendly interface built with **Streamlit**
-  Categorical data is handled automatically using one-hot encoding
-  Real-time data visualizations using **Matplotlib**  
-  Three key visualizations included:
  - **Bar chart** of car counts by manufacturer
  - **Line chart** of average price by production year
  - **Scatter plot** showing price vs. mileage

---

##  Technologies Used

- Python
- The main programming language used to write your code.
- It handles everything: building the app, processing the data, training the model, etc.
- Streamlit
- A UI (User Interface) library.
- It allows you to turn any Python script into an interactive web app, without needing HTML or JavaScript.
- In your code: you use it to build interactive elements like dropdowns, number inputs, and display results/visualizations.
- Pandas
- A data analysis and cleaning library.
- It helps you read CSV files, clean and filter columns/rows, and manipulate data easily.
- In your code: it reads the dataset, drops unnecessary columns, and converts strings to numbers.
- Scikit-learn
- A machine learning library.
- You use it to train a model that can predict car prices based on the inputs.
- RandomForestRegressor is a powerful model that uses multiple decision trees (a forest) to make accurate predictions.
- Matplotlib
- A library used for plotting charts.
- It’s used alongside Seaborn in your code to visualize insights like price vs. year, top manufacturers, etc.
- Pickels file
- A file format used to save a trained model so you don't have to re-train it every time you run the app.
- It makes your app faster and more efficient.
- In your code: you save the model once it's trained, and load it later when needed.

## Code & Datasets
- [price.py](https://github.com/A-lehlouh/Car-Price-Prediction-App/blob/main/price.py)
- [datasets](https://github.com/A-lehlouh/Car-Price-Prediction-App/blob/main/car_price_prediction%20(Autosaved).csv)

---

## Contact
If you have any questions, feel free to reach out via GitHub Issues or email: [AbdulrahmanLehlouh@gmail.com]

## License
This project is open-source and available under the MIT License.
