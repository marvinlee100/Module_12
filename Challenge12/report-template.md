# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

We were provided with a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. We took the data and split it into training and testing sets. We then created a Logistic Regression model with the original data. Afterwards, we predicted a Logistic Regression model with resampled training data. The purpose of the analysis is to compare the results of the two machine learning models and make a recommendation as to which model to use going forward. 

The financial information in the lending_data.csv file consists of the following elements: 
* loan_size
* interest_rate
* borrower_income
* debt_to_income
* num_of_accounts
* derogatory_marks
* total_debt
* loan_status

When spliting the data into training and testing sets, we had to separate the data into labels and features. The y varaible a.k.a the target variable, is the loan status. The X variable or the features consisted of all the other columns in the dataset. The loan_status is a binary variable that would either be a 0 or a 1. A value of '0' indicated a healthy loan whereas a value of '1' indicated a high-risk loan. Given the features, a prediction of either a '0' or a '1' would appear under the loan_status column. We would check the balance fo the labels variable (y) by using the value_counts function. The original dataset showed a total of 75,036 '0's and 2,500 '1's. 

The stages of the machine learning process are fairly straightforward. We would prepare the data by reading it into a DataFrame. The X and y labels would be created as we separated the data. The data would then be split into training and testing datasets by using the train_test_split function from sklearn.model_selection, a module in the scikit-learn library. A Logistic Regression model would then be created with the original data. We would fit a logistic regression model by using X_train and y_train. These two arrays are a subset of the data features and target variable, respectively, used for training the model. Afterwards, we'd make and save the prediction using the testing data and the fitted model. Finally, we would evaluate the model's performance by calculating the accuracy score of the model, generating a confusion matrix and printing the classification report. 

We would go onto predict a logistic regression model with resampled training data. Using the RandomOverSampler module from the imbalanced-learn library, we would instantiate the random oversampler model and fit the original training data to the random oversampler model. Note that for the y_resampled variable, we got a value count of 56,271 O's and 1's. Using the LogisticRegression classifier and the resampled data, we'd go onto fit the model and make predictions. We would then evaluate the model's performance the same way. 

## Results


Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
  
The balanced accuracy score for Model 1 is 0.9520479. An accuracy score is used to assess performance; however, in an imbalanced class, the accuracy score can prove misleading. Balanced accuracy, on the other hand, is a metric that takes into account the imbalance in the dataset by calculating the average of the recall values for each class. It is calculated as the average of the recall values for each class, i.e., the sum of the recall values divided by the number of classes. Balanced accuracy provides a more accurate measure of a model's performance on imbalanced datasets, as it gives equal weight to each class regardless of its size. In this case, credit risk poses a classification problem that's inherently imbalanced. This is because healthy loans easily outnumber risky loans. With our confusion matrix, we group our model's predictions according to whether the model accurately predicted the categories and it showed the model was significantly accurately in detecting the true positives and true negatives. A balanced accuracy score of 0.9520479 indicates that the model is performing very well at correctly classifying all classes, on average, and is able to account for any class imbalance in the dataset.

Precision is a performance metric used in machine learning to evaluate the quality of a model's predictions. It measures the proportion of true positive predictions (i.e., correctly predicted positive examples) among all positive predictions made by the model. The precision values for healthy and high-risk loans are 1.00 and 0.85, respectively. 

In the context of healthy and high-risk loans, precision values of 1.00 for healthy loans and 0.85 for high-risk loans mean the following:

Precision of 1.00 for healthy loans: This means that when the model predicts a loan as "healthy," it is always correct. In other words, out of all the loans that the model predicted as "healthy," there were no false positives (i.e., loans that were actually high-risk but were predicted as healthy).

Precision of 0.85 for high-risk loans: This means that when the model predicts a loan as "high-risk," it is correct 85% of the time. In other words, out of all the loans that the model predicted as "high-risk," 15% of them were actually healthy loans (i.e., false positives).

Overall, precision values provide insights into the accuracy of a model's predictions for a particular class. A precision of 1.00 is ideal, as it indicates that the model is making no false positive predictions for that class. However, a precision of 0.85 for high-risk loans may indicate that the model needs further improvement to reduce the number of false positive predictions.

Recall, also known as sensitivity or true positive rate, is another performance metric used in machine learning to evaluate the quality of a model's predictions. It measures the proportion of true positive predictions among all actual positive examples.The recall values for healthy and high-risk loans are 0.99 and 0.91, respectively. 

In the context of healthy and high-risk loans, recall values of 0.99 for healthy loans and 0.91 for high-risk loans mean the following:

Recall of 0.99 for healthy loans: This means that the model correctly identifies 99% of all healthy loans in the dataset. In other words, out of all the actual healthy loans in the dataset, the model correctly predicted 99% of them as healthy.

Recall of 0.91 for high-risk loans: This means that the model correctly identifies 91% of all high-risk loans in the dataset. In other words, out of all the actual high-risk loans in the dataset, the model correctly predicted 91% of them as high-risk.

Overall, recall values provide insights into the model's ability to correctly identify all positive examples in a dataset for a particular class. A recall of 1.0 is ideal, as it indicates that the model is able to correctly identify all positive examples. However, a recall of 0.91 for high-risk loans may indicate that the model needs further improvement to correctly identify all high-risk loans in the dataset.
 



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
  
The balanced accuracy score for Model 2 is 0.9936781. The confusion matrix showed a higher number of true positives and true negatives. The precision values for healthy and risky loans are 1.00 and 0.84, respectively. The precision values for Model 1 and 2 are basically the same. The recall values for both types of loans are 0.99. This means the model correctly identifies 99% of all healthy and risky loans in the dataset. This 99% recall value is so close to the ideal 100%. 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

The model that appears to perofrm better is Model 2. Model 2 has a higher balanced accuracy score and recall value. The accuracy score and recall value are so close to the ideal 1.0. The confusion matrix for Model 2 shows a noticeable improvement from Model 1. It would be my recommendation that Model 2 be used as importance goes to the ability to accurately predict the '1's (risky loans). While Model 2 has a slightly lower precision value and may produce more false positives (15% vs 16%), it is not disqualifying that healthy loans get flagged a bit more often as unhealthy loans. The main objective is to correctly identify risky loans. 
