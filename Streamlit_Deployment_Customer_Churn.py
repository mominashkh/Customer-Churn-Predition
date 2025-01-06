import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(page_title='Churn Prediction App', layout='wide')

# Title
st.title('Customer Churn Prediction App')

# File Upload
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())
    st.write("### Dataset Statistics")
    st.write(data.describe())

    # Data Preprocessing
    data['Churn'] = data['Churn'].astype('category').cat.codes
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify=y)
    
    scaler = StandardScaler()
    X_train.iloc[:, :17] = scaler.fit_transform(X_train.iloc[:, :17])
    X_test.iloc[:, :17] = scaler.transform(X_test.iloc[:, :17])

    # Model Selection
    st.sidebar.title('Model Selection')
    model_option = st.sidebar.selectbox('Choose a Model', ['KNN', 'Decision Tree'])

    if model_option == 'KNN':
        st.write('## KNN Model')
        k_values = [5, 7, 9, 13, 15]
        param_grid = {'n_neighbors': k_values}
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_knn = grid_search.best_estimator_
        
        st.write(f"Best k: {grid_search.best_params_['n_neighbors']}")
        results = pd.DataFrame(grid_search.cv_results_)
        
        # Plot Accuracy vs k
        st.write('### KNN Accuracy vs k')
        fig, ax = plt.subplots()
        ax.plot(results['param_n_neighbors'], results['mean_test_score'])
        ax.set_xlabel('Number of Neighbors (k)')
        ax.set_ylabel('Accuracy')
        st.pyplot(fig)
        
        knn_predictions = best_knn.predict(X_test)
        st.write('### KNN Performance Metrics')
        st.write('Confusion Matrix:', confusion_matrix(y_test, knn_predictions))
        st.write('Precision:', precision_score(y_test, knn_predictions, pos_label=1))
        st.write('Recall:', recall_score(y_test, knn_predictions, pos_label=1))
        st.write('F1 Score:', f1_score(y_test, knn_predictions, pos_label=1))

    if model_option == 'Decision Tree':
        st.write('## Decision Tree Model')
        dt = DecisionTreeClassifier(random_state=123)
        dt.fit(X_train, y_train)
        
        # Plot Tree
        st.write('### Decision Tree Plot')
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt, feature_names=X.columns, class_names=['False', 'True'], filled=True, ax=ax)
        st.pyplot(fig)
        
        dt_predictions = dt.predict(X_test)
        st.write('### Decision Tree Performance Metrics')
        st.write('Confusion Matrix:', confusion_matrix(y_test, dt_predictions))
        st.write('Precision:', precision_score(y_test, dt_predictions, pos_label=1))
        st.write('Recall:', recall_score(y_test, dt_predictions, pos_label=1))
        st.write('F1 Score:', f1_score(y_test, dt_predictions, pos_label=1))
        
        # Adjust Probability Threshold
        dt_prob = pd.DataFrame(dt.predict_proba(X_test), columns=['False', 'True'])
        dt_prob['pred_class'] = np.where(dt_prob['False'] > 0.85, 0, 1)
        dt_prob['pred_class2'] = np.where(dt_prob['True'] > 0.15, 1, 0)
        
        st.write('### Adjusted Threshold Metrics')
        st.write('Confusion Matrix:', confusion_matrix(y_test, dt_prob['pred_class2']))
        st.write('Precision:', precision_score(y_test, dt_prob['pred_class2'], pos_label=1))
        st.write('Recall:', recall_score(y_test, dt_prob['pred_class2'], pos_label=1))
        st.write('F1 Score:', f1_score(y_test, dt_prob['pred_class2'], pos_label=1))

else:
    st.info('Please upload a CSV file to continue.')

st.write('---')
st.write('Built with ❤️ using Streamlit')
