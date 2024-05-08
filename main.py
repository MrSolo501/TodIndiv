import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Функция для вычисления рекомендуемого количества кластеров
def compute_num_clusters(data):
    max_clusters = min(len(data), 10)
    wcss = []
    for i in range(1, max_clusters):
        clustering = AgglomerativeClustering(n_clusters=i)
        labels = clustering.fit_predict(data)
        cluster_centers = np.array([data[labels == j].mean(axis=0) for j in range(i)])
        wcss.append(((data - cluster_centers[labels]) ** 2).sum())
    num_clusters = np.argmin(np.gradient(wcss)) + 1
    return num_clusters

# Загрузка данных
@st.cache_data  # новая функция кэширования
def load_data(file):
    return pd.read_csv(file)

# Заголовок приложения
st.title('Приложение для иерархической кластеризации')

# Загрузка данных
uploaded_file = st.file_uploader("Загрузить CSV файл", type=['csv'])
if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write('### Данные:')
    st.write(data.head())

    # Предобработка данных
    # Удаление столбцов с нечисловыми значениями
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) < len(data.columns):
        st.warning("Удалены столбцы с нечисловыми значениями.")
        data = data[numeric_columns]

    # Проверка наличия пропущенных значений
    if data.isnull().sum().sum() > 0:
        # Заполнение пропущенных значений средними значениями
        imputer = SimpleImputer(strategy='mean')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Нормализация данных
    #scaler = StandardScaler()
    #data_scaled = scaler.fit_transform(data)
    data_scaled = preprocessing.MinMaxScaler().fit_transform(data)

    # Вывод нормализованных данных
    st.write('### Нормализованные данные:')
    st.write(data_scaled)

    # Выбор целевых столбцов
    target_columns = st.multiselect('Выберите целевые столбцы:', data.columns)

    # Определение рекомендуемого количества кластеров
    num_clusters = compute_num_clusters(data_scaled)

    # Ввод количества кластеров пользователем
    num_clusters_custom = st.text_input(
        'Введите количество кластеров (или оставьте пустым для автоматического выбора):', '')

    # Проверка на корректность введенного значения
    try:
        num_clusters_custom = int(num_clusters_custom)
    except ValueError:
        st.error('Пожалуйста, введите допустимое количество кластеров.')

    if isinstance(num_clusters_custom, int) and num_clusters_custom > 1:
        num_clusters = num_clusters_custom

    if isinstance(num_clusters, int) and num_clusters > 1:
        # Иерархическая кластеризация
        mergings = linkage(data_scaled, method="complete", metric='euclidean')

        # Визуализация дендрограммы
        fig, ax = plt.subplots(figsize=(10, 8))  # Изменение размера дендрограммы
        dendrogram(mergings, ax=ax, truncate_mode='lastp', p=num_clusters)
        ax.set_title('Дендрограмма иерархической кластеризации')
        ax.set_xlabel('Индекс выборки')
        ax.set_ylabel('Расстояние')
        st.pyplot(fig)

        # Выполнение кластеризации на преобразованных данных
        clustering = AgglomerativeClustering(n_clusters=num_clusters)
        labels = clustering.fit_predict(data_scaled)

        # Визуализация кластеров на двумерном графике
        if data_scaled.shape[1] >= 2:
            fig, ax = plt.subplots()
            for cluster in range(num_clusters):
                cluster_data = pd.DataFrame(data_scaled[labels == cluster], columns=data.columns)
                ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1])

            # Визуализация центроидов кластеров
            centroids = np.zeros((num_clusters, data_scaled.shape[1]))
            for cluster in range(num_clusters):
                cluster_data = pd.DataFrame(data_scaled[labels == cluster], columns=data.columns)
                centroids[cluster] = cluster_data.mean(axis=0)
                ax.scatter(centroids[cluster, 0], centroids[cluster, 1], marker='x', color='black')
                #вывод центров кластеров
                st.write(f'Центр кластера {cluster + 1}: {centroids[cluster]}')

            ax.set_title('Визуализация кластеров с центроидами')
            ax.set_xlabel('Признак 1')
            ax.set_ylabel('Признак 2')
            st.pyplot(fig)

        # Визуализация кластеров на основе размера кластера
        cluster_sizes = np.bincount(labels)
        if len(cluster_sizes) > 1:
            fig, ax = plt.subplots()
            ax.bar(range(num_clusters), cluster_sizes)
            ax.set_title('Размеры кластеров')
            ax.set_xlabel('Кластер')
            ax.set_ylabel('Размер')
            st.pyplot(fig)

        # Вывод таблиц для каждого кластера
        for cluster in range(num_clusters):
            st.write(f'### Кластер {cluster + 1}:')
            cluster_data = data[labels == cluster]
            st.write(cluster_data)

    elif num_clusters_custom != '':
        st.error('Пожалуйста, введите допустимое количество кластеров (больше 1).')
