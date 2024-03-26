from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_multilabel_classification
from sklearn.datasets import make_s_curve
from sklearn.datasets import make_swiss_roll
from sklearn.datasets import make_low_rank_matrix
from sklearn.datasets import make_sparse_coded_signal
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(
   page_title="Synthetic Dataset Generation",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

col1, col2, col3, col4 = st.columns(4)


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
    plt.show(fig)
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

    st.subheader('Gaussian Quantiles')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_gaussian_quantiles
    import matplotlib.pyplot as plt

    # Generate Gaussian quantiles data
    X, y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_gaussian_quantiles')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate Gaussian quantiles data
    X, y = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=3, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_gaussian_quantiles')
    st.pyplot(fig)

    st.divider()

    st.subheader('Swiss Roll')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Generate Swiss Roll data
    X, y = make_swiss_roll(n_samples=1000, random_state=42)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Generated Data with make_swiss_roll')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate Swiss Roll data
    X, y = make_swiss_roll(n_samples=1000, random_state=42)

    # Plot the generated data
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Generated Data with make_swiss_roll')
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
    plt.show(fig)
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

    st.divider()

    st.subheader('Hastie 10_2')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_hastie_10_2
    import matplotlib.pyplot as plt

    # Generate Hastie 10_2 data
    X, y = make_hastie_10_2(n_samples=1000, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_hastie_10_2')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate Hastie 10_2 data
    X, y = make_hastie_10_2(n_samples=1000, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_hastie_10_2')
    st.pyplot(fig)

    st.divider()

    st.subheader('Low-Rank Matrix')
    st.code('''
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_low_rank_matrix

    # Generate Low-Rank Matrix data
    data = make_low_rank_matrix(n_samples=100, n_features=100, effective_rank=10, noise=0.1, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_title('Generated Low-Rank Matrix')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Sample')
    st.pyplot(fig)
    ''')
    
    # Generate Low-Rank Matrix data
    data = make_low_rank_matrix(n_samples=100, n_features=100, effective_rank=10, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Value')
    ax.set_title('Generated Low-Rank Matrix')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Sample')
    st.pyplot(fig)
    


with col3:
    st.subheader('Circle')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_circles
    import matplotlib.pyplot as plt

    # Generate circles data
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_circles')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate circles data
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_circles')
    st.pyplot(fig)

    st.divider()

    st.subheader('Multilabel Classification')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_multilabel_classification
    import matplotlib.pyplot as plt

    # Generate Multilabel Classification data
    X, y = make_multilabel_classification(n_samples=1000, n_features=2, n_classes=3, random_state=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_multilabel_classification')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate Multilabel Classification data
    X, y = make_multilabel_classification(n_samples=1000, n_features=2, n_classes=3, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_multilabel_classification')
    st.pyplot(fig)

    st.divider()


with col4:
    st.subheader('Blob')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate blobs data
    X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_blobs')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate blobs data
    X, y = make_blobs(n_samples=1000, centers=3, cluster_std=1.0, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Generated Data with make_blobs')
    st.pyplot(fig)

    st.divider()

    st.subheader('S Curve')
    st.code('''
    import streamlit as st
    from sklearn.datasets import make_s_curve
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Generate S Curve data
    X, y = make_s_curve(n_samples=1000, random_state=42)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Generated Data with make_s_curve')

    # Display the plot in Streamlit app
    plt.show(fig)
    ''')
    
    # Generate S Curve data
    X, y = make_s_curve(n_samples=1000, random_state=42)

    # Plot the generated data
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Generated Data with make_s_curve')
    st.pyplot(fig)