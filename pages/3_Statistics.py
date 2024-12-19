import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.set_page_config(
   page_title="Statistics",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)
st.subheader('Statistics terms')
col1, col2, col3, col4 = st.columns(4)


with col1:
    st.subheader('Mean')
    st.latex(r'''
        {\small\begin{equation*}
            \overline{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
        \end{equation*}}
        ''', help="Accuracy is the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances. It provides an overall measure of how well the classifier performs across all classes.")
    
    st.code('''
            
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    def average(numbers: list) -> float:
        return sum(numbers) / len(numbers)
    
    sepal_length = df['sepal length (cm)'].to_list()
    avg_value = average(sepal_length)


    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(sepal_length)
    ax.axvline(x=avg_value, color='r', linestyle='-')
    ax.set_xlabel('Sepal length (cm)')
    ax.set_title('Plot Mean')
    ''')

    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    def average(numbers: list) -> float:
        return sum(numbers) / len(numbers)
    
    sepal_length = df['sepal length (cm)'].to_list()
    avg_value = average(sepal_length)


    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(sepal_length)
    ax.axvline(x=avg_value, color='r', linestyle='-')
    ax.set_xlabel('Sepal length (cm)')
    ax.set_title('Plot of Sepal Length')

    st.pyplot(fig)

    st.divider()


with col2:
    st.subheader('Median')
    st.markdown('''
    For odd values: $(n+1)/2$ index of element in array.

    For even values: average of values by indexes: $(n/2)$ and $(n/2 + 1)$.
    ''')
    st.code('''
            
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    
    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    def average(numbers: list) -> float:
        return sum(numbers) / len(numbers)
    
    sepal_length = df['sepal length (cm)'].to_list()
    avg_value = average(sepal_length)


    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(sepal_length)
    ax.axvline(x=avg_value, color='r', linestyle='-')
    ax.set_xlabel('Sepal length (cm)')
    ax.set_title('Plot of Sepal Length')
    ''')

    dataset = load_iris()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    def median(numbers: list) -> float | int:
        n = len(numbers)
        isodd = n % 2

        if isodd:
            index = int((n+1)/2)
            return numbers[index-1]
        else:
            index_f = int(n/2) - 1
            index_s = int(n/2)

            return (numbers[index_f] + numbers[index_s]) / 2
    
    sepal_length = df['sepal length (cm)'].to_list()


    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(sepal_length)
    ax.axvline(median(sepal_length), color='r', linestyle='-')
    ax.set_xlabel('Sepal length (cm)')
    ax.set_title('Plot Median')

    st.pyplot(fig)

    st.divider()


with col3:
    st.subheader('Mode')
    st.markdown('''
    Mode is useful for categorical features analysis
    ''')
    st.code('''
            
    
    import matplotlib.pyplot as plt
    
    def get_mode(items: list):
        count_values = {}

        for item in items:
            value: str = str(item)
            
            if value in count_values:
                count_values[value] += 1
            else:
                count_values[value] = 1
        
        max_count = max(count_values.values())
        mode = []
        
        for key, value in count_values.items():
            if value == max_count:
                mode.append(key)

        return {
            'mode': mode, 
            'count_values': count_values
            }

    x = ['apple', 'apple', 'orange', 'orange', 'banana', 'orange', 'apple']
    mode_attrs = get_mode(x)
    mode, count_values = mode_attrs['mode'], mode_attrs['count_values']

    x, y = zip(*count_values.items())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(x, y)
    ax.set_title('Plot Mode')
    ''')

    def get_mode(items: list):
        count_values = {}

        for item in items:
            value: str = str(item)
            
            if value in count_values:
                count_values[value] += 1
            else:
                count_values[value] = 1
        
        max_count = max(count_values.values())
        mode = []
        
        for key, value in count_values.items():
            if value == max_count:
                mode.append(key)

        return {
            'mode': mode, 
            'count_values': count_values
            }

    x = ['apple', 'apple', 'orange', 'orange', 'banana', 'orange', 'apple']
    mode_attrs = get_mode(x)
    mode, count_values = mode_attrs['mode'], mode_attrs['count_values']

    x, y = zip(*count_values.items())

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.barh(x, y)
    ax.set_title('Plot Mode')

    st.pyplot(fig)

    st.divider()


with col4:
    st.subheader('Covariance')

    st.markdown('''
        Covariance is a measure of joint variability of two random variables.
    ''')

    st.latex(r'''
        {\small\begin{equation*}
            cov(X,Y) = \frac{1}{n}\sum_{i=0}^n{(x_i - \overline{x})(y_i - \overline{y})}
        \end{equation*}}
        ''',
        help='''
        Covariance is a measure of joint variability of two random variables.
        If the greater values of variable mainly correspond with greater values of the other variable,
        and the same holds for the lesser values (that is, the variables tend to show similar behaviour), the covariance is positive. 
        In the opposite case, when the greater values of one variable mainly correspond to
        the fewer values of the other - the covariance is negative.The sign of the covarience, therefor,
        shows tendency in the linear relationship between the variables.''')


    data_url = "http://lib.stat.cmu.edu/datasets/boston"

    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    X = raw_df[1]
    Z = raw_df[2]

    def get_covariance(x: list, y: list):
        x_var_mean = np.array(x).mean()
        y_var_mean = np.array(y).mean()

        multple_vars = 0
        
        for i in range(len(x)):
            x_delta = x[i] - x_var_mean
            y_delta = y[i] - y_var_mean

            multple_vars += x_delta * y_delta
        
        return multple_vars / len(x)
    
    m = raw_df.cov().style.background_gradient(cmap='coolwarm')
    m

    st.divider()

    st.subheader('Correlation')
    st.markdown('''
        Correlation
    ''')

    st.latex(r'''
        {\small\begin{equation*}
            r = \frac{\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2 \sum_{i=1}^{n} (y_i - \overline{y})^2}}
        \end{equation*}}
        ''',
        help='''''')

    c = raw_df.corr().style.background_gradient(cmap='coolwarm')
    c