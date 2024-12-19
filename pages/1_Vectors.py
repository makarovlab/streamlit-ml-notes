import streamlit as st
import matplotlib.pyplot as plt

class VectorsPage:
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
        st.markdown("hello vectors", help="")

def main():
    VectorsPage.config()
    VectorsPage.render()

if __name__ == "__main__":
    main()