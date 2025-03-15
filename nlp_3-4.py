import os
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
from datasets import load_dataset

from nltk.corpus import stopwords
from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from collections import Counter
from wordcloud import WordCloud

ag_news = load_dataset("ag_news")
X_ag = ag_news["train"]["text"] + ag_news["test"]["text"]
y_ag = np.concatenate([ag_news["train"]["label"], ag_news["test"]["label"]])

imdb = load_dataset("imdb")
X_imdb = imdb["train"]["text"] + imdb["test"]["text"]
y_imdb = np.concatenate([imdb["train"]["label"], imdb["test"]["label"]])

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.PorterStemmer()
stop_words = set(stopwords.words("english"))

def visualize_top_words(word_counts, top_n):
    top_words = dict(word_counts.most_common(top_n))
    wordcloud = WordCloud(
        background_color='white',
        width=800,
        height=400,
        colormap='viridis'
    ).generate_from_frequencies(top_words)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Топ-{top_n} самых частых слов')
    plt.show()

def preprocess(text, method='none', stopword=False, part='none', top_n=0):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    if part == 'noun':
        tagged_tokens = pos_tag(tokens)
        tokens = [word for word, tag in tagged_tokens if tag.startswith('N')]

    if method == 'lemma':
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    elif method == 'stem':
        tokens = [stemmer.stem(token) for token in tokens]

    if stopword:
        word_counts = Counter(tokens)
        top_words = {word for word, _ in word_counts.most_common(top_n)} 
        tokens = [token for token in tokens if token not in top_words]
        
    return ' '.join(tokens)


def evaluate(X, y, preprocess_method, vectorize_method, dataset_name, stopword, part, use_lsa=False, k=None):
    processed = [preprocess(text, preprocess_method, stopword, part, top_n=10) for text in X]
    
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier()
    }
    
    results = {}
    
    X_train, X_test, y_train, y_test = train_test_split(
        processed, 
        y, 
        test_size=0.2, 
        random_state=42
    )
    
    for name, model in models.items():
        model_id = f"{dataset_name}_{name}_{preprocess_method}_{vectorize_method}_{stopword}_{part}_{use_lsa}_{k}".replace(' ', '_')
        model_path = f'models/{model_id}.pkl'
        
        pipeline_steps = []
 
        if vectorize_method == 'tfidf':
            pipeline_steps.append(('vectorizer', TfidfVectorizer()))
        else:
            pipeline_steps.append(('vectorizer', CountVectorizer()))

        if use_lsa and k:
            pipeline_steps.append(('lsa', TruncatedSVD(n_components=k)))
        
        pipeline_steps.append(('classifier', model))

        pipeline = Pipeline(pipeline_steps)
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                pipeline = pickle.load(f)
        else:
            pipeline.fit(X_train, y_train)
            os.makedirs("models", exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)

        y_pred = pipeline.predict(X_test)
        metric = f1_score(y_test, y_pred, average='macro' if dataset_name == 'ag_news' else 'binary')
        results[name] = {'F1 Score': metric}
    
    return results

datasets = {
    'ag_news': (X_ag, y_ag),
    'imdb': (X_imdb, y_imdb)
}

preprocess_methods = ['none', 'lemma', 'stem']
vectorize_methods = ['frequency', 'tfidf']
part_speech = ['none', 'noun']
custom_stopwords = [False, True]
lsa_params = [
    {'use_lsa': False, 'k': None}, 
    {'use_lsa': True, 'k': 3},    
    {'use_lsa': True, 'k': 10},
    {'use_lsa': True, 'k': 50}
]
def generate_wordclouds(dataset_name, X, preprocess_method, top_n=10):
    sample_texts = X[:100] 
    processed = [
        preprocess(text, method=preprocess_method, part='noun') 
        for text in sample_texts
    ]
    all_tokens = ' '.join(processed).split()
    word_counts = Counter(all_tokens)
    visualize_top_words(word_counts, top_n=top_n)
    plt.savefig(f'{dataset_name}_{preprocess_method}_wordcloud.png')
    plt.close()

for dataset_name in datasets:
    X, y = datasets[dataset_name]
    generate_wordclouds(dataset_name, X, 'lemma', top_n=10)

results = []
for dataset_name in datasets:
    X, y = datasets[dataset_name]
    for config in lsa_params:
        for preproc in preprocess_methods:
            for vec in vectorize_methods:
                for part in part_speech:
                    for stopword in custom_stopwords:
                        print(f"Processing {dataset_name}, LSA={config['use_lsa']}, k={config['k']}, preproc={preproc}")
                        res = evaluate(
                            X, y, 
                            preprocess_method=preproc,
                            vectorize_method=vec,
                            dataset_name=dataset_name,
                            stopword=stopword,
                            part=part,
                            **config
                        )
                        for model, scores in res.items():
                            results.append({
                                'Dataset': dataset_name,
                                'LSA': config['use_lsa'],
                                'k': config['k'],
                                'Preprocess': preproc,
                                'Vectorizer': vec,
                                'Part Speech': part,
                                'Stopwords': stopword,
                                'Model': model,
                                **scores
                            })

df_results = pd.DataFrame(results) 
df_results.to_csv('results.csv', index=False)
print(f"Топ 5 лучших конфигураций:\n{df_results.nlargest(5, 'F1 Score')}")
