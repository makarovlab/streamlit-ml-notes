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

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Combinatorics')
    st.markdown("#### Permutations:")
    st.latex(r'''
        {\small\begin{equation*}
            P(n) = n!
        \end{equation*}}
        ''', help="")
    
    st.markdown("#### Combinations:")
    st.latex(r'''
        {\small\begin{equation*}
            C(n, k) = \binom{n}{k} = \frac{n!}{k!(n-k)!}
        \end{equation*}}
        ''', help="")
    
    st.markdown("#### Arrangments:")
    st.latex(r'''
        {\small\begin{equation*}
            A(n, k) = \frac{n!}{(n-k)!}
        \end{equation*}}
        ''', help="")
    