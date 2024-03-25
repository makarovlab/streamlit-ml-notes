import streamlit as st

st.set_page_config(
   page_title="Metrics for Classification",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)



st.subheader('Confusion Matrix')
conf_matrix_col1, conf_matrix_col2 = st.columns(2)

with conf_matrix_col1:
    st.markdown('''
        1. True Positive (TP): The number of instances of positive class correctly predicted by the classifier.
        2. True Negative (TN): The number of instances of negative class correctly predicted by the classifier.
        3. False Positive (FP) (Type I error): The number of instances of negative class incorrectly predicted as positive by the classifier.
        4. False Negative (FN) (Type II error): The number of instances of positive class incorrectly predicted as negative by the classifier.

        ||Predicted Negative|Predicted Positive|
        |-|-|-|
        |Actual Negative|TN|FP|
        |Actual Positive|FN|TP|
    ''')

with conf_matrix_col2:
    st.image('./pages/images/conf_matrix.png', width=300)


col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Accuracy')
    st.latex(r'''
        {\small\begin{equation*}
            Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
        \end{equation*}}
        ''', help="Accuracy is the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances. It provides an overall measure of how well the classifier performs across all classes.")
    st.code('''
        from sklearn.metrics import accuracy_score
        ...
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        ''')
    
    st.markdown('''
    **Advantages:**
    1.  Intuitive Interpretation;
    2.  Simple Calculation;
    3.  Applicability to Binary and Multiclass Problems.

    **Disadvantages:**
    1.  Sensitivity to Class Imbalance;
    2.  Inability to Distinguish Errors;
    3.  Not Suitable for Skewed Datasets;
    4.  Doesn't Consider Confidence Levels.
    ''')
    
    st.divider()

    st.subheader('F1 Score')
    st.latex(r'''
        {\small\begin{equation*}
            F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
        \end{equation*}}
        ''', help="The F1 score is the harmonic mean of precision and recall. It provides a balance between precision and recall, giving equal weight to both metrics. ")
    st.code('''
        from sklearn.metrics import f1_score
        ...
        f1 = f1_score(y_test, y_pred)
        print("F1 Score:", f1)
        ''')
    
    st.markdown('''
    **Advantages:**
    1.  Harmonic Mean of Precision and Recall
    2.  Useful for Binary and Multiclass Problems
    3.  Provides Balance Between Precision and Recall
    4.  Sensitivity to Both False Positives and False Negatives

    **Disadvantages:**
    1.  Dependence on Threshold
    2.  Difficulty in Interpretation
    3.  Limited Insight into Trade-offs
    4.  Not Suitable for All Applications
    ''')
    
    st.divider()

with col2:
    st.subheader('Precision')
    st.latex(r'''
        {\small\begin{equation*}
            Precision = \frac{TP}{TP + FP}
        \end{equation*}}
        ''', help="Precision measures the proportion of positive instances among the instances that the classifier predicted as positive. It's a measure of the classifier's ability to avoid false positives. ")
    st.code('''
        from sklearn.metrics import precision_score
        ...
        precision = precision_score(y_test, y_pred)
        print("Precision:", precision)
        ''')
    
    st.markdown('''
    **Advantages:**
    1.  Focus on Relevant Instances
    2.  Useful for Imbalanced Datasets
    3.  Sensitive to False Positive Rate
    4.  Complementary to Recall

    **Disadvantages:**
    1.  Ignorance of False Negatives
    2.  Dependence on Threshold
    3.  Incomplete Picture
    4.  Limited Applicability to Multiclass Problems
    ''')
    
    st.divider()


with col3:
    st.subheader('Recall')
    st.latex(r'''
        {\small\begin{equation*}
            Recall = \frac{TP}{TP + FN}
        \end{equation*}}
        ''', help="Recall, also known as Sensitivity or True Positive Rate (TPR), measures the proportion of actual positive instances that were correctly predicted by the classifier. It's a measure of the classifier's ability to find all the positive instances.")
    st.code('''
        from sklearn.metrics import recall_score
        ...
        recall = recall_score(y_test, y_pred)
        print("Recall:", recall)
        ''')
    
    st.markdown('''
    **Advantages:**
    1.  Focus on Identifying Positive Instances
    2.  Useful for Imbalanced Datasets
    3.  Sensitive to False Negative Rate
    4.  Complementary to Precision

    **Disadvantages:**
    1.  Ignorance of False Positives
    2.  Dependence on Threshold
    3.  Incomplete Picture
    4.  Trade-off with Precision
    ''')
    
    st.divider()


st.header("Iris dataset example")
st.code('''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a classifier (e.g., K-Nearest Neighbors)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
''')