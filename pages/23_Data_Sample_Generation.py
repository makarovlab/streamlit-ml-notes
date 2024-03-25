import streamlit as st
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

st.set_page_config(
   page_title="Synthetic Dataset Generation",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

col1, col2, col3 = st.columns(3)


with col1:
    st.subheader('Classification')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_classification
    import matplotlib.pyplot as plt
    # Generate a synthetic dataset with 2 classes and 2 informative features
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

    # Plot the synthetic dataset and store the figure in a variable
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Synthetic Dataset - 2 Informative Features')

    # Display the plot in Streamlit app
    st.pyplot(fig)
    ''')
    
    
    # Generate a synthetic dataset with 2 classes and 2 informative features
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42)

    # Plot the synthetic dataset and store the figure in a variable
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, marker='o', edgecolors='k')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Synthetic Dataset - 2 Informative Features')

    # Display the plot in Streamlit app
    st.pyplot(fig)
    
    st.divider()


with col2:
    st.subheader('Regression')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_regression
    import matplotlib.pyplot as plt

    # Generate synthetic regression data with 1 feature
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

    # Plot the synthetic regression data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color='blue', marker='o', edgecolors='k')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.set_title('Synthetic Regression Dataset')

    # Display the plot in Streamlit app
    st.pyplot(fig)
    ''')
    
    # Generate synthetic regression data with 1 feature
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

    # Plot the synthetic regression data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color='blue', marker='o', edgecolors='k')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.set_title('Synthetic Regression Dataset')

    # Display the plot in Streamlit app
    st.pyplot(fig)