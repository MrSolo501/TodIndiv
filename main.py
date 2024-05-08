import streamlit as st
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Функция для вычисления рекомендуемого количества кластеров
def compute_num_clusters(data, numeric_columns):
    max_clusters = min(len(numeric_columns), 10)
    wcss = []
    for i in range(1, max_clusters):
        clustering = AgglomerativeClustering(n_clusters=i)
        labels = clustering.fit_predict(data[numeric_columns])
        cluster_centers = np.array([data[numeric_columns][labels == j].mean(axis=0) for j in range(i)])
        wcss.append(((data[numeric_columns] - cluster_centers[labels]) ** 2).sum())
    num_clusters = np.argmin(np.gradient(wcss)) + 1
    return num_clusters

# Заголовок приложения
st.title('Приложение для иерархической кластеризации')

# Загрузка данных
uploaded_file = st.file_uploader("Загрузить CSV файл", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('### Данные:')
    st.write(data.head())

    # Выбор целевых столбцов
    target_columns = st.multiselect('Выберите целевые столбцы:', data.columns)

    # Проверка, что выбран хотя бы один целевой столбец
    if not target_columns:
        st.error('Пожалуйста, выберите как минимум один целевой столбец.')
    else:
        # Фильтрация числовых столбцов
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Вычисление рекомендуемого количества кластеров
        num_clusters = compute_num_clusters(data, numeric_columns)
        st.write(f'### Рекомендуемое количество кластеров: {num_clusters}')

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
            # Извлечение данных только для выбранных целевых столбцов
            selected_data = data[target_columns]

            # Преобразование категориальных переменных в фиктивные (dummy) переменные
            data_encoded = pd.get_dummies(selected_data)

            # Иерархическая кластеризация
            mergings = linkage(data_encoded, method="complete", metric='euclidean')

            # Визуализация дендрограммы
            fig, ax = plt.subplots(figsize=(10, 8))  # Изменение размера дендрограммы
            dendrogram(mergings, ax=ax, truncate_mode='lastp', p=num_clusters)
            ax.set_title('Дендрограмма иерархической кластеризации')
            ax.set_xlabel('Индекс выборки')
            ax.set_ylabel('Расстояние')
            st.pyplot(fig)

            # Выполнение кластеризации на преобразованных данных
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(data_encoded)

            # Визуализация кластеров на двумерном графике
            if data_encoded.shape[1] >= 2:  # Проверка, что у нас есть как минимум 2 признака для визуализации
                fig, ax = plt.subplots()
                for cluster in range(num_clusters):
                    cluster_data = data_encoded[labels == cluster]
                    ax.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1])

                # Визуализация центроидов кластеров
                centroids = np.zeros((num_clusters, data_encoded.shape[1]))
                for cluster in range(num_clusters):
                    cluster_data = data_encoded[labels == cluster]
                    centroids[cluster] = cluster_data.mean(axis=0)
                    ax.scatter(centroids[cluster, 0], centroids[cluster, 1], marker='x', color='black')

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
                cluster_data = data.loc[labels == cluster]
                st.write(cluster_data)

        elif num_clusters_custom != '':
            st.error('Пожалуйста, введите допустимое количество кластеров (больше 1).')
