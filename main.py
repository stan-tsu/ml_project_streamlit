import pandas as pd
import numpy as np
import streamlit as st
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action='ignore', category=FutureWarning)
from catboost import CatBoostRegressor

pd.set_option('display.max_columns', None)

# Функция предобработки данных
def preprocess_data(df):
    """Предобработка данных - та же логика, что применялась к тренировочным данным"""
    df_processed = df.copy()
    
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageQual', 'GarageCond', 'GarageType', 'GarageFinish',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                 'MasVnrType']

    # Заполняем пропуски 'None' для указанных колонок
    for col in none_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('None')

    # Заполняем LotFrontage медианой по районам
    if 'LotFrontage' in df_processed.columns and 'Neighborhood' in df_processed.columns:
        df_processed['LotFrontage'] = df_processed.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))

    # Заполняем нулями
    if 'GarageYrBlt' in df_processed.columns:
        df_processed['GarageYrBlt'] = df_processed['GarageYrBlt'].fillna(0)
    if 'MasVnrArea' in df_processed.columns:
        df_processed['MasVnrArea'] = df_processed['MasVnrArea'].fillna(0)

    # Заполняем модой
    mode_cols = ['MSZoning', 'KitchenQual', 'Electrical', 'Exterior1st', 'Exterior2nd', 
                 'SaleType', 'Functional', 'Utilities']

    for col in mode_cols:
        if col in df_processed.columns:
            mode_val = df_processed[col].mode()
            if len(mode_val) > 0:
                df_processed[col] = df_processed[col].fillna(mode_val[0])
    
    return df_processed

# Загрузка и подготовка тренировочных данных
@st.cache_data
def load_and_prepare_training_data():
    """Загружает и подготавливает тренировочные данные"""
    try:
        test = pd.read_csv('test.csv')
        train = pd.read_csv('train.csv')
        
        train_labels = train['SalePrice']
        y_train = np.log1p(train_labels)
        
        # Объединяем train и test для консистентного кодирования
        full = pd.concat([train.drop(columns=['SalePrice']), test], axis=0).reset_index(drop=True)
        
        # Предобработка
        full = preprocess_data(full)
        
        # Кодирование категориальных признаков
        cat_cols = full.select_dtypes(include='object').columns.tolist()
        full_encoded = pd.get_dummies(full, columns=cat_cols, drop_first=True)
        
        X_train = full_encoded.iloc[:len(train), :]
        
        return X_train, y_train, full_encoded.columns.tolist(), train
    except Exception as e:
        st.error(f"Ошибка при загрузке тренировочных данных: {str(e)}")
        return None, None, None, None

# Обучение модели
@st.cache_resource
def train_model():
    """Обучает модель на тренировочных данных"""
    X_train, y_train, feature_columns, train_data = load_and_prepare_training_data()
    if X_train is None:
        return None, None, None
    
    model = CatBoostRegressor(verbose=0, n_estimators=500, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    return model, feature_columns, train_data

# Функция для страницы с EDA
def show_eda(train):
    st.header('Анализ распределения и зависимостей SalePrice')
    
    # Гистограмма
    st.subheader('Распределение SalePrice')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(train['SalePrice'], kde=True, bins=30, color='blue', ax=ax1)
    ax1.set_title('Распределение SalePrice', fontsize=16)
    ax1.set_xlabel('SalePrice', fontsize=14)
    ax1.set_ylabel('Частота', fontsize=14)
    st.pyplot(fig1)
    plt.close(fig1)
    
    st.write("Гистограмма показывает распределение цен на дома (SalePrice). Распределение имеет правостороннюю асимметрию (скошено вправо), "
             "что означает наличие значительного количества домов с относительно низкой стоимостью и небольшого числа дорогих объектов. "
             "KDE (ядровая оценка плотности) подтверждает эту асимметрию. Для дальнейшего анализа может потребоваться логарифмическое "
             "преобразование данных, чтобы приблизить распределение к нормальному.")
    
    # Boxplot
    st.subheader('Boxplot для SalePrice')
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=train['SalePrice'], color='orange', ax=ax2)
    ax2.set_title('Boxplot для SalePrice', fontsize=16)
    ax2.set_xlabel('SalePrice', fontsize=14)
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.write("Boxplot показывает, что распределение цен на жилье ('SalePrice') имеет правую асимметрию: медиана находится в районе "
             "100000–200000, но присутствуют выбросы в области высоких цен (до 700000), что указывает на наличие дорогих объектов, "
             "значительно отклоняющихся от типичного диапазона. Большинство данных сосредоточено в нижней части графика, а "
             "интерквартильный размах относительно узкий.")
    
    # Scatterplots
    st.subheader('Зависимость SalePrice от числовых признаков')
    numerical_features = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea']
    
    for feature in numerical_features:
        if feature in train.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=train[feature], y=train['SalePrice'], alpha=0.6, color='green', ax=ax)
            ax.set_title(f'Зависимость SalePrice от {feature}', fontsize=16)
            ax.set_xlabel(feature, fontsize=14)
            ax.set_ylabel('SalePrice', fontsize=14)
            st.pyplot(fig)
            plt.close(fig)
            
            if feature == 'GrLivArea':
                st.write("**GrLivArea:**\nНаблюдается положительная линейная зависимость между площадью жилых помещений (GrLivArea) и ценой продажи (SalePrice). "
                      "С увеличением GrLivArea стоимость дома в целом растёт. Однако есть несколько выбросов: дома с большой площадью, но относительно "
                      "низкой ценой, что может указывать на дополнительные факторы, влияющие на стоимость.\n")
            elif feature == 'TotalBsmtSF':
                st.write("**TotalBsmtSF:**\nНаблюдается положительная корреляция между общей площадью подвала (TotalBsmtsF) и ценой продажи (SalePrice). "
                      "Чем больше площадь подвала, тем выше стоимость дома, что логично, так как большие подвалы увеличивают полезную площадь жилья.\n"
                      "Однако есть выбросы.\n")
            elif feature == '1stFlrSF':
                st.write("**1stFlrSF:**\nНаблюдается положительная линейная зависимость: с увеличением площади первого этажа (1stFlrSF) растёт и цена дома (SalePrice). "
                      "Это логично, так как большая жилая площадь обычно повышает стоимость недвижимости.\n"
                      "Выбросы: Некоторые дома с очень большой площадью (3000+ кв. футов) имеют непропорционально низкую цену, что может объясняться "
                      "плохим состоянием, или расположением в неудачном месте.\n")
            elif feature == 'GarageArea':
                st.write("**GarageArea:**\nНаблюдается умеренная положительная зависимость между площадью гаража и ценой дома: в целом, с увеличением GarageArea "
                      "растет и SalePrice, но связь не такая четкая, как для жилой площади.\n"
                      "Выбросы: Есть несколько дорогих домов с маленькими гаражами, что может указывать на премиальное расположение или уникальные "
                      "характеристики (например, историческая ценность). Также присутствуют недорогие дома с большими гаражами — вероятно, это старые "
                      "объекты или требующие ремонта.")

# Функция для корреляционного анализа
def show_correlation_analysis(train):
    st.header('Анализ корреляций и распределения по районам')
    
    # Оставим только числовые признаки
    numeric_feats = train.select_dtypes(include=['int64', 'float64'])
    
    # Корреляционная матрица
    st.subheader('Корреляция с SalePrice')
    corr_matrix = numeric_feats.corr()
    
    # Топ коррелирующих признаков
    top_corr_features = corr_matrix['SalePrice'][abs(corr_matrix['SalePrice']) > 0.5].sort_values(ascending=False)
    
    # Тепловая карта
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix.loc[top_corr_features.index, top_corr_features.index],
                annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, 
                vmin=-1, vmax=1, ax=ax1)
    ax1.set_title("Тепловая карта корреляции (основные характеристики с ценой продажи)", fontsize=14)
    st.pyplot(fig1)
    plt.close(fig1)
    
    st.write("""
    **Анализ корреляции:**
    - На цену жилья (SalePrice) сильнее всего влияют:
      - Качество постройки (OverallQual, 0.79)
      - Жилая площадь (GrLivArea, 0.71) 
      - Размер гаража (GarageCars, 0.64)
    - Некоторые признаки дублируют информацию (например, GarageArea и GarageCars с корреляцией 0.88)
    - Для моделирования рекомендуется выбрать наиболее значимые и независимые факторы.
    """)
    
    # График по районам
    st.subheader('Распределение цен по районам')
    
    # Сортировка районов по медианной цене
    order = train.groupby("Neighborhood")["SalePrice"].median().sort_values().index
    
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.boxplot(x="Neighborhood", y="SalePrice", data=train, order=order, ax=ax2)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_title("Распределение цены продажи по районам", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.write("""
    **Анализ распределения по районам:**
    - Значительные различия в ценах между районами
    - От бюджетных вариантов (левая часть) до премиальных (правая часть)
    - Широкий разброс цен в некоторых районах
    - Наличие выбросов свидетельствует о неоднородности рынка
    - Присутствуют как стандартные, так и аномально дорогие/дешёвые объекты
    """)

# Функция для страницы "О проекте"
def show_about_project():
    st.title('🏠 О проекте: Предсказание цен на недвижимость')
    
    # Заголовок и описание
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #1f77b4; text-align: center;">📊 Анализ и предсказание цен на дома</h2>
        <p style="font-size: 18px; text-align: center; color: #333;">
            Комплексный проект машинного обучения для предсказания стоимости жилой недвижимости
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Команда проекта
    st.header('👥 Команда проекта')
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #e1f5fe; border-radius: 10px;">
            <h3>👨‍💻 Станислав</h3>
            <p>Data Scientist</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #e8f5e8; border-radius: 10px;">
            <h3>👨‍💻 Вадим</h3>
            <p>ML Engineer</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #fff3e0; border-radius: 10px;">
            <h3>👨‍💻 Сергей</h3>
            <p>Data Analyst</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #fce4ec; border-radius: 10px;">
            <h3>👨‍💻 Денис</h3>
            <p>ML Specialist</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Источник данных
    st.header('📂 Источник данных')
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 10px 0;">
        <strong>Датасет:</strong> train.csv с платформы <a href="https://www.kaggle.com" target="_blank">Kaggle</a><br>
        <strong>Конкурс:</strong> House Prices - Advanced Regression Techniques<br>
        <strong>Размер:</strong> 1460 записей, 81 признак
    </div>
    """, unsafe_allow_html=True)
    
    # Этапы работы
    st.header('🔄 Этапы работы над проектом')
    
    # Этап 1
    st.subheader('1️⃣ Анализ и разведка данных')
    st.markdown("""
    - **Распределение задач:** Разделили все колонки между участниками команды
    - **Анализ качества данных:** Выявление пропусков, нулей, аномальных значений
    - **Анализ типов данных:** Определение числовых и категориальных признаков
    - **Планирование кодирования:** Выбор методов обработки для каждого признака
    """)
    
    # Этап 2
    st.subheader('2️⃣ Предобработка данных')
    st.markdown("""
    **Методы заполнения пропусков:**
    - 🔢 **Заполнение нулями:** Для числовых признаков (GarageYrBlt, MasVnrArea)
    - 📊 **Заполнение медианой:** Для LotFrontage с группировкой по районам
    - 📈 **Заполнение модой:** Для категориальных признаков (MSZoning, KitchenQual)
    - 📝 **Заполнение 'None':** Для категориальных признаков, где отсутствие = признак
    """)
    
    # Этап 3
    st.subheader('3️⃣ Визуализация и анализ')
    st.markdown("""
    - 📊 **Графики до обработки:** Анализ исходного распределения данных
    - 🔧 **Функции очистки:** Создание переиспользуемых функций предобработки
    - 📈 **Графики после кодирования:** Валидация результатов обработки
    - 🎯 **Корреляционный анализ:** Выявление связей между признаками
    """)
    
    # Этап 4
    st.subheader('4️⃣ Моделирование')
    st.markdown("""
    **Стратегия:** Параллельная работа команды для поиска оптимального решения
    
    **Использованные алгоритмы:**
    - 🎯 **Lasso Regression:** Линейная регрессия с L1-регуляризацией
    - 🌳 **Random Forest Regression:** Ансамбль деревьев решений
    - 🚀 **CatBoost Regressor:** Градиентный бустинг (финальная модель)
    """)
    
    # Этап 5
    st.subheader('5️⃣ Отбор признаков и валидация')
    st.markdown("""
    - 🎯 **Feature Selection:** Использование F1 Score для отбора 50 лучших признаков
    - ✅ **Кросс-валидация:** Проверка устойчивости модели на разных выборках
    - 📊 **Оценка качества:** Метрики RMSE, MAE для регрессии
    """)
    
    # Результаты
    st.header('🏆 Результаты')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #e8f5e8; border-radius: 10px;">
            <h2 style="color: #4caf50; margin: 0;">715</h2>
            <p style="margin: 5px 0;"><strong>Место на Kaggle</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #e3f2fd; border-radius: 10px;">
            <h2 style="color: #2196f3; margin: 0;">CatBoost</h2>
            <p style="margin: 5px 0;"><strong>Лучшая модель</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #fff3e0; border-radius: 10px;">
            <h2 style="color: #ff9800; margin: 0;">50</h2>
            <p style="margin: 5px 0;"><strong>Отобранных признаков</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Технологии
    st.header('🛠️ Используемые технологии')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Библиотеки Python:**
        - 🐼 **Pandas** - обработка данных
        - 🔢 **NumPy** - численные вычисления
        - 📊 **Matplotlib/Seaborn** - визуализация
        - 🤖 **Scikit-learn** - машинное обучение
        - 🚀 **CatBoost** - градиентный бустинг
        """)
    
    with col2:
        st.markdown("""
        **Инструменты:**
        - 📓 **Jupyter Notebook** - разработка
        - 🌐 **Streamlit** - веб-приложение
        - 📈 **Kaggle** - платформа соревнований
        - 🔄 **Git** - контроль версий
        - 🐍 **Python 3.8+** - язык программирования
        """)
    
    # Выводы
    st.header('💡 Ключевые выводы')
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #28a745;">
        <ul>
            <li><strong>Качество данных критично:</strong> Правильная обработка пропусков значительно улучшает результат</li>
            <li><strong>Feature Engineering важен:</strong> Отбор признаков с помощью F1 Score повысил точность модели</li>
            <li><strong>Ансамбли работают:</strong> CatBoost показал лучшие результаты среди всех протестированных моделей</li>
            <li><strong>Командная работа эффективна:</strong> Параллельное тестирование разных подходов ускорило поиск решения</li>
            <li><strong>Валидация обязательна:</strong> Кросс-валидация помогла избежать переобучения</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    

# Функция для страницы с предсказаниями
def show_prediction_page(final_model, feature_columns):
    st.title('Predict sales prices')
    st.write('Загрузи csv-файл для предсказания цен')
    
    # Загрузка файла для предсказания
    uploaded_df = st.file_uploader('Загрузка', type='csv')
    if uploaded_df is not None:
        df = pd.read_csv(uploaded_df)
        st.write("Загруженные данные:")
        st.write(df.head())
        
        if st.button('Предсказать цены'):
            try:
                # Предобработка загруженных данных
                df_processed = preprocess_data(df)
                
                # Кодирование категориальных признаков
                cat_cols = df_processed.select_dtypes(include='object').columns.tolist()
                df_encoded = pd.get_dummies(df_processed, columns=cat_cols, drop_first=True)
                
                # Приводим к тем же колонкам, что были при обучении
                df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)
                
                # Предсказание
                y_pred_log = final_model.predict(df_encoded)
                y_pred = np.expm1(y_pred_log)
                
                # Показываем результаты
                st.write("Предсказанные цены:")
                results_df = pd.DataFrame({
                    'Index': range(len(y_pred)),
                    'Predicted_Price': y_pred
                })
                
                # Если есть колонка Id, используем её
                if 'Id' in df.columns:
                    results_df['Id'] = df['Id'].values
                    results_df = results_df[['Id', 'Predicted_Price']]
                
                st.write(results_df)
                
                # Возможность скачать результаты
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Скачать результаты в CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Ошибка при предсказании: {str(e)}")
                st.error("Убедитесь, что структура загруженного файла соответствует тренировочным данным")
    else:
        st.info("Загрузите CSV файл для получения предсказаний")

# Главная функция приложения
def main():
    st.set_page_config(page_title="House Price Prediction", layout="wide", page_icon="🏠")
    
    # Боковая панель для навигации
    st.sidebar.title("🏠 Навигация")
    page = st.sidebar.selectbox("Выберите страницу", [
        "🔮 Предсказание цен", 
        "📊 Анализ данных (EDA)", 
        "🔗 Корреляционный анализ",
        "📋 О проекте"
    ])
    
    # Обучаем модель только если не на странице "О проекте"
    if page != "📋 О проекте":
        with st.spinner('Загрузка данных и обучение модели...'):
            final_model, feature_columns, train_data = train_model()
        
        if final_model is None or feature_columns is None:
            st.error("Не удалось обучить модель. Проверьте наличие файлов train.csv и test.csv")
            st.stop()
        
        st.sidebar.success("✅ Модель успешно обучена!")
    
    # Отображение выбранной страницы
    if page == "🔮 Предсказание цен":
        show_prediction_page(final_model, feature_columns)
    elif page == "📊 Анализ данных (EDA)":
        if train_data is not None:
            show_eda(train_data)
        else:
            st.error("Не удалось загрузить тренировочные данные для анализа")
    elif page == "🔗 Корреляционный анализ":
        if train_data is not None:
            show_correlation_analysis(train_data)
        else:
            st.error("Не удалось загрузить тренировочные данные для анализа")
    elif page == "📋 О проекте":
        show_about_project()

if __name__ == "__main__":
    main()