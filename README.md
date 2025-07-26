Предсказание риска сердечного приступа

Небольшое веб-приложение на FastAPI, которое принимает CSV с данными пациентов, запускает обученный пайплайн и выводит список ID людей с высоким риском сердечного приступа.

Структура проекта<br/>

m1/<br/>
├── app.py              # FastAPI-приложение <br/>
├── model.py            # Класс HeartRiskModel: загрузка пайплайна + фича-инжиниринг + predict()<br/>
├── test.py             # Простой скрипт для проверки эндпоинта /health и отправки тестового CSV<br/>
├── best_model.pkl      # Сериализованный GridSearchCV.best_estimator_<br/>
├── templates/          # HTML-шаблоны<br/>
│   ├── start_form.html # Форма загрузки CSV<br/>
│   └── res_form.html   # Отображение списка ID с высоким риском<br/>
└── tmp/                # Временная папка для сохраняемых загруженных CSV<br/>

Установка и запуск

1. Клонировать (или скачать) папку m1 целиком
2. Перейти в неё: cd m1
3. Установить зависимости (рекомендуется виртуальное окружение): pip install fastapi uvicorn joblib pandas scikit-learn catboost jinja2
4. Запустить сервер: python -m uvicorn app:app --reload
5. Открыть в браузере: http://127.0.0.1:8000/


Как это работает
1. app.py
    Инициализация
    Загружается пайплайн из best_model.pkl в класс HeartRiskModel, задаётся порог threshold=0.5
    Эндпоинты
        GET /health → возвращает 200 OK
        GET / → отрисовывает шаблон start_form.html
        POST /predict →
            Получает файл под ключом file
            Сохраняет во временную папку tmp/
            Передаёт DataFrame в HeartRiskModel.predict()
            Если в ходе препроцессинга или предсказания возникает ошибка (отсутствуют столбцы и т.п.), перенаправляет на /?error=…
            Формирует список high_risk ID пациентов с model.predict_proba ≥ threshold
            Отображает этот список в шаблоне res_form.html
2. model.py
Класс HeartRiskModel:
    TOP_FEATURES — 14 признаков
        Топ-10 по SHAP:
        bmi, income, systolic_blood_pressure, triglycerides,
        sedentary_hours_per_day, cholesterol, exercise_hours_per_week,
        heart_rate, age, stress_level
        Плюс 4 новых:
        lifestyle_risk, active_score, pulse_pressure, age_group

    Методы
        __init__(model_path, threshold)
        — загружает пайплайн из model_path
        _feature_engineering(df)
        — создаёт:
            lifestyle_risk = smoking + alcohol_consumption + obesity
            active_score = exercise_hours_per_week – sedentary_hours_per_day
            pulse_pressure = systolic_blood_pressure – diastolic_blood_pressure
            age_group = pd.qcut(age, 4, labels=False)

        predict(df)
            Приводит колонки к нижнему регистру и snake_case
            Вызывает _feature_engineering
            Берёт только TOP_FEATURES и идёт через сохранённый пайплайн
            Возвращает три списка: ID всех пациентов,  бинарные предсказания по порогу,  вероятности

3. test.py

Простой скрипт, проверяющий:
    GET http://127.0.0.1:8000/health → 200 OK
    POST http://127.0.0.1:8000/predict с любым CSV → получение HTML-страницы с результатами



Обучение модели
    Данные heart_train.csv и heart_test.csv:
        Размеченный train (с таргетом), незамеченный test
        28 столбцов: антропометрия, привычки, давление, хронические, биохимия + id + целевой heart_attack_risk_binary

    EDA + FE
        Графики (hist, boxplot) для количественных.
        Баланс/дискретность двоичных
        Категории (gender, diet)
        Удалены выбросы heart_rate (>0.2)
        Удалены пропуски в двоичных фичах (241 запись)

    Корреляция
    — нет сильной мультиколлинеарности (макс до ±0.5)
    — слабая линейная связь с таргетом → нужны нелинейные модели

    Модели
        GridSearchCV (cv=5, scoring=‘f1_weighted’) по пайплайну:
            LogisticRegression + OHE + пасстру (для бинарных + числовых)
            RandomForestClassifier(class_weight='balanced')

        Изначально ROC-AUC ≈0.56, F1_weighted ≈0.58
        Отдельно подбирался CatBoostClassifier(...).fit(..., cat_features=...) → ROC-AUC ≈0.55

    Итог
    — Лучшая модель: RandomForestClassifier с class_weight='balanced' и подбором гиперпараметров по F1_weighted = 0.58 на кросс-валидации
