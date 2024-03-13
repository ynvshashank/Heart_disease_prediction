# Heart_disease_prediction
This code aims to explore and compare the performance of two distinct machine learning models for predicting heart disease based on a given dataset. The dataset contains various attributes related to heart health, such as age, gender, blood pressure, and cholesterol levels.

Firstly, the dataset is loaded and basic exploratory analysis is conducted to understand its structure and characteristics. Then, the data is preprocessed, including splitting it into predictor variables (features) and the target variable (the presence or absence of heart disease).

Subsequently, two different machine learning algorithms are trained and evaluated:

Logistic Regression:

Logistic regression is a linear classification model that predicts the probability of an instance belonging to a particular class.
In this context, logistic regression is trained on the heart disease dataset to learn the relationship between the input features and the presence or absence of heart disease.

Random Forest Classifier:
Random Forest is an ensemble learning method that builds multiple decision trees during training and combines their predictions through voting or averaging to improve accuracy and reduce overfitting.
In this code, a Random Forest classifier is trained on the heart disease dataset using a range of hyperparameters to create a forest of decision trees for predicting heart disease.

Finally, the accuracy scores of these models are calculated and compared to determine which one performs best in terms of accurately predicting the presence or absence of heart disease. The model with the highest accuracy score on the test set is considered the most effective for this particular dataset and task.
