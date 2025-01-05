import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('customer_churn.csv')
print(data.head())
print(data.describe())

# Convert Churn column to categorical
data['Churn'] = data['Churn'].astype('category')

# Encode Churn as 0 and 1 for binary classification
data['Churn'] = data['Churn'].cat.codes

# Split data into training and testing datasets
X = data.drop('Churn', axis=1)
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)

# Standardize numeric columns (excluding target variable)
scaler = StandardScaler()
X_train.iloc[:, :17] = scaler.fit_transform(X_train.iloc[:, :17])
X_test.iloc[:, :17] = scaler.transform(X_test.iloc[:, :17])

# KNN Model
knn = KNeighborsClassifier()
k_values = [5, 7, 9, 13, 15]
param_grid = {'n_neighbors': k_values}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best KNN Model
best_knn = grid_search.best_estimator_
print('Best k:', grid_search.best_params_)

# Plot KNN accuracy
results = pd.DataFrame(grid_search.cv_results_)
plt.plot(results['param_n_neighbors'], results['mean_test_score'])
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. k')
plt.show()

# KNN Predictions
knn_predictions = best_knn.predict(X_test)
print('KNN Confusion Matrix:', confusion_matrix(y_test, knn_predictions))
print('Precision:', precision_score(y_test, knn_predictions, pos_label=1))
print('Recall:', recall_score(y_test, knn_predictions, pos_label=1))
print('F1 Score:', f1_score(y_test, knn_predictions, pos_label=1))

# Decision Tree Model
dt = DecisionTreeClassifier(random_state=123)
dt.fit(X_train, y_train)

# Plot Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['False', 'True'], filled=True)
plt.show()

# Decision Tree Predictions
dt_predictions = dt.predict(X_test)
print('Decision Tree Confusion Matrix:', confusion_matrix(y_test, dt_predictions))
print('Precision:', precision_score(y_test, dt_predictions, pos_label=1))
print('Recall:', recall_score(y_test, dt_predictions, pos_label=1))
print('F1 Score:', f1_score(y_test, dt_predictions, pos_label=1))

# Adjust Probability Threshold
dt_prob = pd.DataFrame(dt.predict_proba(X_test), columns=['False', 'True'])
dt_prob['pred_class'] = np.where(dt_prob['False'] > 0.85, 0, 1)
dt_prob['pred_class2'] = np.where(dt_prob['True'] > 0.15, 1, 0)

print('Adjusted Threshold Confusion Matrix:', confusion_matrix(y_test, dt_prob['pred_class2']))
print('Precision:', precision_score(y_test, dt_prob['pred_class2'], pos_label=1))
print('Recall:', recall_score(y_test, dt_prob['pred_class2'], pos_label=1))
print('F1 Score:', f1_score(y_test, dt_prob['pred_class2'], pos_label=1))
