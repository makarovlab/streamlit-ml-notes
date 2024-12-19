import streamlit as st
import matplotlib.pyplot as plt

class MatricesPage:
    @staticmethod
    def config():
        st.set_page_config(
            page_title="Statistics",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @staticmethod
    def title(title="Matrix Properties Cheat Sheet"):
        st.title(title)

    @staticmethod
    def render():
        col1, col2 = st.columns(2)

        # Layout with 3 columns for better organization
        col1, col2, col3 = st.columns(3)

        # Matrix Addition
        with col1:
            st.subheader("Matrix Addition")
            st.latex(r'''
            C = A + B
            \quad \text{where} \quad C = [c_{ij}], \quad c_{ij} = a_{ij} + b_{ij}
            ''')

        # Matrix Subtraction
        with col2:
            st.subheader("Matrix Subtraction")
            st.latex(r'''
            C = A - B
            \quad \text{where} \quad C = [c_{ij}], \quad c_{ij} = a_{ij} - b_{ij}
            ''')

        # Matrix Scalar Multiplication
        with col3:
            st.subheader("Scalar Multiplication")
            st.latex(r'''
            \alpha A = [\alpha a_{ij}]
            ''')

        # Matrix Multiplication
        with col1:
            st.subheader("Matrix Multiplication")
            st.latex(r'''
            C = A \cdot B \quad \text{where} \quad c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}
            ''')

        # Matrix Transpose
        with col2:
            st.subheader("Matrix Transposition")
            st.latex(r'''
            A^T = [a_{ji}]
            ''')

        # Matrix Inversion
        with col3:
            st.subheader("Matrix Inversion")
            st.latex(r'''
            A \cdot A^{-1} = A^{-1} \cdot A = I
            ''')

        # Determinant
        with col1:
            st.subheader("Determinant")
            st.latex(r'''
            \text{det}(A) = \left| \begin{matrix} a & b \\ c & d \end{matrix} \right| = ad - bc
            ''')

        # Eigenvalues and Eigenvectors
        with col2:
            st.subheader("Eigenvalues & Eigenvectors")
            st.latex(r'''
            A v = \lambda v
            ''')

        # Trace of a Matrix
        with col3:
            st.subheader("Trace")
            st.latex(r'''
            \text{tr}(A) = \sum_{i=1}^{n} a_{ii}
            ''')

        # Rank of a Matrix
        with col1:
            st.subheader("Rank")
            st.latex(r'''
            \text{rank}(A) = \min(\text{number of rows}, \text{number of columns})
            ''')

        # Symmetric Matrix
        with col2:
            st.subheader("Symmetric Matrix")
            st.latex(r'''
            A = A^T
            ''')

        # Orthogonal Matrix
        with col3:
            st.subheader("Orthogonal Matrix")
            st.latex(r'''
            Q^T Q = Q Q^T = I
            ''')

        # Diagonal Matrix
        with col1:
            st.subheader("Diagonal Matrix")
            st.latex(r'''
            D = [d_{ij}] \quad \text{where} \quad d_{ij} = 0 \quad \text{for} \quad i \neq j
            ''')

        # Identity Matrix
        with col2:
            st.subheader("Identity Matrix")
            st.latex(r'''
            I = [\delta_{ij}] \quad \text{where} \quad \delta_{ij} = 1 \quad \text{if} \quad i = j
            ''')


def main():
    MatricesPage.config()
    MatricesPage.render()

if __name__ == "__main__":
    main()