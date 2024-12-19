import streamlit as st
import matplotlib.pyplot as plt

class MathLanguagePage:
    @staticmethod
    def config():
        st.set_page_config(
            page_title="Statistics",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @staticmethod
    def render():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("### **Basic Operators**")
            st.latex(r"""
            \begin{align*}
            + & - \text{Addition} \\
            - & - \text{Subtraction} \\
            \times & - \text{Multiplication} \\
            \div & - \text{Division} \\
            \cdot & - \text{Dot Product} \\
            \pm & - \text{Plus-Minus} \\
            \mp & - \text{Minus-Plus}
            \end{align*}
            """)

            st.markdown("### **Greek Letters**")
            st.latex(r"""
            \begin{align*}
            \alpha & - \text{Alpha} \\
            \beta & - \text{Beta} \\
            \gamma & - \text{Gamma} \\
            \delta & - \text{Delta} \\
            \epsilon & - \text{Epsilon} \\
            \theta & - \text{Theta} \\
            \lambda & - \text{Lambda} \\
            \pi & - \text{Pi} \\
            \sigma & - \text{Sigma} \\
            \phi & - \text{Phi} \\
            \omega & - \text{Omega}
            \end{align*}
            """)

            st.markdown("### **Vector Symbols**")
            st.latex(r"""
            \begin{align*}
            \vec{v} & : \text{Vector } v \\
            |\vec{v}| & : \text{Magnitude of Vector } v \\
            \hat{i}, \hat{j}, \hat{k} & : \text{Unit Vectors} \\
            \vec{a} \cdot \vec{b} & : \text{Dot Product} \\
            \vec{a} \times \vec{b} & : \text{Cross Product} \\
            \nabla \cdot \vec{A} & : \text{Divergence of Vector Field} \\
            \nabla \times \vec{A} & : \text{Curl of Vector Field} \\
            \begin{bmatrix} a \\ b \\ c \end{bmatrix} & : \text{3D Vector} \\
            \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} & : \text{n-dimensional Vector}
            \end{align*}
            """)
        
        with col2:
            st.markdown("### **Comparison Symbols**")
            st.latex(r"""
            \begin{align*}
            = & - \text{Equal} \\
            \neq & - \text{Not Equal} \\
            \approx & - \text{Approximately Equal} \\
            < & - \text{Less Than} \\
            > & - \text{Greater Than} \\
            \leq & - \text{Less Than or Equal To} \\
            \geq & - \text{Greater Than or Equal To}
            \end{align*}
            """)

            st.markdown("### **Miscellaneous Symbols**")
            st.latex(r"""
            \begin{align*}
            \infty & : \text{Infinity} \\
            \propto & : \text{Proportional To} \\
            \therefore & : \text{Therefore} \\
            \because & : \text{Because} \\
            \angle & : \text{Angle} \\
            \sum & : \text{Sum} \\
            \% & : \text{Percentage}
            \end{align*}
            """)

            st.markdown("### **Discrete Math Symbols**")
            st.latex(r"""
            \begin{align*}
            \land & : \text{Logical AND} \\
            \lor & : \text{Logical OR} \\
            \neg & : \text{Logical NOT} \\
            \implies & : \text{Implication} \\
            \iff & : \text{If and Only If} \\
            \oplus & : \text{Exclusive OR (XOR)} \\
            \sum_{i=1}^{n} a_i & : \text{Finite Sum} \\
            \prod_{i=1}^{n} a_i & : \text{Finite Product} \\
            \bmod & : \text{Modulo Operator} \\
            \equiv & : \text{Equivalent} \\
            
            \end{align*}
            """)
        
        with col3:
            st.markdown("### **Set and Logic Symbols**")
            st.latex(r"""
            \begin{align*}
            \in & - \text{Element Of} \\
            \notin & - \text{Not an Element Of} \\
            \subset & - \text{Subset} \\
            \subseteq & - \text{Subset or Equal} \\
            \cap & - \text{Intersection} \\
            \cup & - \text{Union} \\
            \emptyset & - \text{Empty Set} \\
            \forall & - \text{For All} \\
            \exists & - \text{Exists}
            \end{align*}
            """)

            st.markdown("### **Math Symbols for Sets**")
            st.latex(r'''
            \begin{align*}
            \mathbb{N} & : \text{Set of Natural Numbers} \\
            \mathbb{Z} & : \text{Set of Integers} \\
            \mathbb{Q} & : \text{Set of Rational Numbers} \\
            \mathbb{R} & : \text{Set of Real Numbers} \\
            \mathbb{C} & : \text{Set of Complex Numbers}
            \end{align*}
            ''')

            st.markdown("### **Matrix Symbols**")
            st.latex(r"""
            \begin{align*}
            A^T & : \text{Transpose of Matrix} \\
            A^{-1} & : \text{Inverse of Matrix} \\
            \det(A) & : \text{Determinant of Matrix} \\
            \text{tr}(A) & : \text{Trace of Matrix} \\
            I_n & : \text{Identity Matrix of Size } n \\
            \begin{bmatrix} a & b \\ c & d \end{bmatrix} & : \text{Example of a 2x2 Matrix} \\
            \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} & : \text{3x3 Identity Matrix} \\
            \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix} & : \text{3x2 Matrix Example}
            \end{align*}
            """)

            st.markdown("### **Linear equations**")
            st.latex(r'''
            \left\{
            \begin{aligned}
            3x + 2y &= 5 \\
            2x - y &= 1
            \end{aligned}
            \right.
            ''')

        
        with col4:
            st.markdown("### **Calculus Symbols**")
            st.latex(r"""
            \begin{align*}
            \int & : \text{Integral} \\
            \iint & : \text{Double Integral} \\
            \oint & : \text{Closed Integral} \\
            \sum & : \text{Summation} \\
            \prod & : \text{Product} \\
            \nabla & : \text{Gradient} \\
            \partial & : \text{Partial Derivative} \\
            \frac{dy}{dx} & : \text{Derivative}
            \end{align*}
            """)

            st.markdown("### **Limits Symbols**")
            st.latex(r"""
            \begin{align*}
            \lim_{x \to a} f(x) & : \text{Limit as } x \text{ approaches } a \\
            \lim_{n \to \infty} a_n & : \text{Limit as } n \text{ approaches infinity} \\
            \limsup_{n \to \infty} a_n & : \text{Limit superior} \\
            \liminf_{n \to \infty} a_n & : \text{Limit inferior} \\
            \end{align*}
            """)

            st.markdown("### **Miscellaneous Symbols**")
            st.latex(r"""
            \begin{align*}
            \infty & : \text{Infinity} \\
            \propto & : \text{Proportional To} \\
            \therefore & : \text{Therefore} \\
            \because & : \text{Because} \\
            \angle & : \text{Angle} \\
            \sum & : \text{Sum} \\
            \% & : \text{Percentage}
            \end{align*}
            """)
        
            st.markdown("### **Logarithm Symbols**")
            st.latex(r"""
            \begin{align*}
            \log(x) & : \text{Logarithm Base 10} \\
            \ln(x) & : \text{Natural Logarithm (Base } e) \\
            \log_b(x) & : \text{Logarithm of } x \text{ Base } b \\
            \log_2(x) & : \text{Binary Logarithm (Base 2)} \\
            \log_e(x) & : \text{Natural Logarithm (Base } e)
            \end{align*}
            """)


def main():
    MathLanguagePage.config()
    MathLanguagePage.render()

if __name__ == "__main__":
    main()