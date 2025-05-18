import numpy as np
import pandas as pd
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from datasets import load_dataset
import nltk

lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = word_tokenize(text.lower())
    processed = []
    for token in tokens:
        if token in stop_words:
            continue
        lemma = lemmatizer.lemmatize(token)
        processed.append(lemma)
    return ' '.join(processed)

def load_and_preprocess(dataset_name):
    dataset = load_dataset(dataset_name)
    X_train = [preprocess(text) for text in dataset['train']['text']]
    X_test = [preprocess(text) for text in dataset['test']['text']]
    y_train = np.array(dataset['train']['label'])
    y_test = np.array(dataset['test']['label'])
    
    return X_train, X_test, y_train, y_test

tfidf_params = {
    'max_df': [0.5, 0.7, 0.9],    # Максимальная доля документов, где может встречаться слово
    'min_df': [1, 3, 5],           # Минимальное количество документов для слова
    'ngram_range': [(1,1), (1,2)], # Используемые n-граммы
    'max_features': [5000, 10000], # Максимальное количество признаков
    'sublinear_tf': [True, False], # Сублинейное масштабирование TF
    'stop_words': ['english']      # Английские стоп-слова
}

model_params = {
    'RandomForest': {
        'clf__n_estimators': [100, 200],      # Количество деревьев
        'clf__max_depth': [None, 50],         # Максимальная глубина
        'clf__min_samples_split': [2, 5]      # Минимум образцов для разделения
    },
    'GBM': {
        'clf__max_depth': [3, 5],             # Глубина деревьев
        'clf__learning_rate': [0.1, 0.05],    # Скорость обучения
        'clf__subsample': [0.8, 1.0]          # Доля образцов для обучения
    },
    'AdaBoost': {
        'clf__estimator__max_depth': [3, 5], # Глубина базовых деревьев
        'clf__n_estimators': [50, 100],       # Количество классификаторов
        'clf__learning_rate': [0.5, 1.0]      # Скорость обучения
    },
    'XGBoost': {
        'clf__max_depth': [3, 5],             # Глубина деревьев
        'clf__learning_rate': [0.1, 0.05],    # Скорость обучения
        'clf__subsample': [0.8, 0.6],         # Доля образцов
        'clf__colsample_bytree': [0.8, 0.6],  # Доля призаков
        'clf__tree_method': ['gpu_hist'],     # Использование GPU
        'clf__gpu_id': [0]                    # ID GPU
    },
    'LogisticRegression': {
        'clf__C': [0.1, 1.0, 10.0],          # Сила регуляризации
        'clf__penalty': ['l2'],              # Тип регуляризации
        'clf__max_iter': [1000]              # Максимум итераций
    }
}

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GBM': GradientBoostingClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    ),
    'LogisticRegression': LogisticRegression(random_state=42)
}

results = []

for dataset_name in ['ag_news', 'imdb']:
    print(f"\n{'='*40}")
    print(f"Начинаем обработку датасета: {dataset_name.upper()}")
    print(f"{'='*40}\n")
    
    X_train, X_test, y_train, y_test = load_and_preprocess(dataset_name)
    
    for model_name in models:
        print(f"\n--- Настройка модели: {model_name} ---")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),  # Шаг векторизации
            ('clf', models[model_name])     # Шаг классификации
        ])

        params = {
            **{f'tfidf__{k}': v for k, v in tfidf_params.items()},
            **model_params.get(model_name, {})
        }

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=10,                 # Уменьшенное количество итераций
            cv=3,                      # 3-кратная кросс-валидация
            scoring='f1_macro',        # Метрика оценки
            n_jobs=-1,                 # Использовать все ядра CPU
            random_state=42,           # Для воспроизводимости
            verbose=1                  # Вывод информации
        )
 
        search.fit(X_train, y_train)
    
        best_params = search.best_params_
        cv_score = search.best_score_
        y_pred = search.predict(X_test)
        test_score = f1_score(y_test, y_pred, average='macro')
    
        print(f"Лучшие параметры:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"CV F1: {cv_score:.4f}")
        print(f"Test F1: {test_score:.4f}")
        print(f"{'-'*40}")

        result = {
            'dataset': dataset_name,
            'model': model_name,
            'cv_score': cv_score,
            'test_score': test_score,
        }
        result.update(best_params)
        results.append(result)

df = pd.DataFrame(results)
df.to_csv('param_res.csv', index=False)
