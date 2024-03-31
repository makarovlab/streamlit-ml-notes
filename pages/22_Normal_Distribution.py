import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def plot_normal_distribution(mu, sigma, num_samples):
    # Generate data
    samples = np.random.normal(mu, sigma, num_samples)
    
    # Graph creation
    plt.figure(figsize=(8, 6))
    plt.hist(samples, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Normal distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu)**2) / (2 * sigma**2))
    plt.plot(x, p, 'k', linewidth=2)
    
    # Добавление меток и заголовка
    plt.title('Нормальное распределение')
    plt.xlabel('Значение')
    plt.ylabel('Плотность вероятности')
    plt.grid(True)
    
    return plt

def main():
    st.title('Визуализация нормального распределения')

    st.latex(r"f(x | \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}")
    
    # Параметры нормального распределения
    mu = st.slider('Выберите среднее значение (μ)', -10.0, 10.0, 0.0, step=0.1)
    sigma = st.slider('Выберите стандартное отклонение (σ)', 0.1, 10.0, 1.0, step=0.1)
    num_samples = st.slider('Выберите количество выборок', 100, 10000, 1000, step=100)
    
    # Создание графика
    plt = plot_normal_distribution(mu, sigma, num_samples)
    st.pyplot(plt)

if __name__ == '__main__':
    main()
