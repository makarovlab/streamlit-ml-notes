import streamlit as st

st.set_page_config(
   page_title="Metrics for Classification",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

col1, col2, col3 = st.columns(3)

with col1:
    col1.subheader('Accuracy')
    col1.latex(r'''
        {\small\begin{equation*}
            Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
        \end{equation*}}
        ''', help="info")
    col1.code('''
        from sklearn.metrics import accuracy_score
        ...
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        ''')
    
    st.divider()

    col1.subheader('F1 Score')
    col1.latex(r'''
        {\small\begin{equation*}
            F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
        \end{equation*}}
        ''', help="info")
    col1.code('''
        from sklearn.metrics import f1_score
        ...
        f1 = f1_score(y_test, y_pred)
        print("F1 Score:", f1)
        ''')
    
    st.divider()

with col2:
    col2.subheader('Precision')
    col2.latex(r'''
        {\small\begin{equation*}
            Precision = \frac{TP}{TP + FP}
        \end{equation*}}
        ''', help="info")
    col2.code('''
        from sklearn.metrics import precision_score
        ...
        precision = precision_score(y_test, y_pred)
        print("Precision:", precision)
        ''')
    
    st.divider()


with col3:
    col3.subheader('Recall')
    col3.latex(r'''
        {\small\begin{equation*}
            Recall = \frac{TP}{TP + FN}
        \end{equation*}}
        ''', help="info")
    col3.code('''
        from sklearn.metrics import recall_score
        ...
        recall = recall_score(y_test, y_pred)
        print("Recall:", recall)
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