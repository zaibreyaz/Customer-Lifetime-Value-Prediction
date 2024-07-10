# Customer Lifetime Value Prediction.

### Overview
This project aims to predict the future value of a customer to a business over the entire duration of their relationship. The prediction model incorporates various factors such as past purchase history, frequency of purchases, and customer demographics. The objective is to enable businesses to make informed decisions about customer retention, marketing strategies, and resource allocation.

### Project Structure
The repository contains the following files and directories:

- `Customer_Lifetime_Value_Prediction.ipynb`: The Jupyter Notebook containing the code and analysis for the project.
- `online-retail-dataset/`: Directory containing the dataset(s) used for the project.
- `models/`: Directory where the trained models and their parameters are saved.
- `README.md`: This file, providing an overview and instructions for the project.
- `requirements.txt`: This file contains all the required libraries to install.

### About Dataset
##### Abstract
A real online retail transaction data set of two years.

##### Data Set Information
This Online Retail II data set contains all the transactions occurring for a UK-based and registered, non-store online retail business between 01/12/2009 and 09/12/2011. The company mainly sells unique all-occasion gift-ware. Many customers of the company are wholesalers.

##### Attribute Information
- `InvoiceNo`: Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'C', it indicates a cancellation.
- `StockCode`: Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.
- `Description`: Product (item) name. Nominal.
- `Quantity`: The quantities of each product (item) per transaction. Numeric.
- `InvoiceDate`: Invoice date and time. Numeric. The day and time when a transaction was generated.
- `UnitPrice`: Unit price. Numeric. Product price per unit in sterling (£).
- `CustomerID`: Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.
- `Country`: Country name. Nominal. The name of the country where a customer resides.

### Installation
To run the project, you need to have the following dependencies installed. The dependencies are listed in the `requirements.txt` file.

You can install the dependencies using pip:
```
pip install -r requirements.txt
```


### Usage
1. `Load the Dataset`: Start by loading the dataset into the Jupyter Notebook. Ensure the dataset is placed in the data/ directory.

2. `Data Preprocessing`: The dataset needs to be preprocessed to handle missing values, encode categorical variables, and normalize numerical features. This step is crucial for preparing the data for model training.

3. `Feature Engineering`: Create additional features that may help improve the model's performance. This includes aggregating purchase history, calculating customer lifetime value, and other relevant metrics.

4. `Model Training`: Train various machine learning models to predict customer lifetime value. This includes linear regression, decision trees, random forests, and gradient boosting.

5. `Model Evaluation`: Evaluate the performance of the trained models using metrics such as Mean Absolute Error (MAE) as our dataset have lots of outliers. Choose the best-performing model for final predictions.

6. `Model Saving and Loading`: 
    - `Saving the Model`: After training the Random Forest Regression model, save it to the `models` folder using `pickle`.

    - `Loading the Model`: Load the saved model from the `models` folder using `pickle`.
        you can load and use the model using the following code: 
        ```
        # Load the model from the 'models' folder
        with open('models/random_forest_regressor_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Use the loaded model to make predictions
        predictions = model.predict(X_test)
        ```

7. `Prediction`: Use the trained model to predict the future value of customers. Save the predictions and model parameters in the models/ directory.

### Results
The results of the project, including model performance metrics, are documented in the Jupyter Notebook. The best-performing model is saved in the `models/` directory and can be used for making future predictions.

### Conclusion
Predicting customer lifetime value is a valuable tool for businesses to understand and optimize customer relationships. By leveraging machine learning techniques, this project demonstrates how to accurately forecast the future value of customers based on their past behavior and demographics.

### Future Work
- `Model Optimization`: Experiment with advanced techniques such as hyperparameter tuning, ensemble methods, and deep learning models to improve prediction accuracy.
- `Feature Expansion`: Incorporate additional features such as customer interactions, social media activity, and external economic factors to enhance the model.
- `Deployment`: Develop a web application or API to make the model accessible to business users for real-time predictions.

### Contributors
- Md Zaib Reyaz (me)
