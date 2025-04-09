import nltk
import numpy as np
import pandas as pd
import string
from datasets import load_dataset
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC  
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def load_data():
    ag_news = load_dataset("ag_news")
    imdb = load_dataset("imdb")
    
    return {
        'ag_news': (
            ag_news["train"]["text"] + ag_news["test"]["text"],
            np.concatenate([ag_news["train"]["label"], ag_news["test"]["label"]])
        ),
        'imdb': (
            imdb["train"]["text"] + imdb["test"]["text"],
            np.concatenate([imdb["train"]["label"], imdb["test"]["label"]])
        )
    }

def preprocess(text, method='stem'):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    
    if method == 'lemma':
        return ' '.join(lemmatizer.lemmatize(t) for t in tokens)
    return ' '.join(stemmer.stem(t) for t in tokens)

def build_pipeline(vectorizer_type, use_lsa, model, k=50):
    steps = []
    
    if vectorizer_type == 'tfidf':
        steps.append(('tfidf', TfidfVectorizer()))
    else:
        steps.append(('count', CountVectorizer()))
    
    if use_lsa and vectorizer_type == 'tfidf':
        steps.append(('lsa', TruncatedSVD(n_components=k)))
    
    steps.append(('clf', model))
    return Pipeline(steps)

def process_combination(comb, model, X_train, y_train, X_test, y_test, dataset_name):
    try:
        pipeline = build_pipeline(
            vectorizer_type=comb['vectorizer'],
            use_lsa=comb['lsa'],
            model=model,
            k=comb.get('k', 50)
        )
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        f1 = f1_score(
            y_test, y_pred,
            average='macro' if dataset_name == 'ag_news' else 'binary'
        )
        
        return {
            'dataset': dataset_name,
            'preproc': comb['preproc'],
            'vectorizer': comb['vectorizer'],
            'lsa': comb['lsa'],
            'k': comb.get('k', 'N/A'),
            'model': model.__class__.__name__,
            'f1_score': round(f1, 4)
        }
        
    except Exception as e:
        print(f"\nError in {comb} with {model.__class__.__name__}: {str(e)}")
        return None

def evaluate_dataset(X, y, dataset_name):
    k_values = [3, 10, 50]
    
    combinations = [
        {'preproc': 'stem', 'vectorizer': 'tfidf', 'lsa': False},
        {'preproc': 'stem', 'vectorizer': 'count', 'lsa': False},
        {'preproc': 'lemma', 'vectorizer': 'tfidf', 'lsa': False},
        {'preproc': 'lemma', 'vectorizer': 'count', 'lsa': False},
        *[{'preproc': 'stem', 'vectorizer': 'tfidf', 'lsa': True, 'k': k} for k in k_values],
        *[{'preproc': 'lemma', 'vectorizer': 'tfidf', 'lsa': True, 'k': k} for k in k_values],
    ]

    with tqdm(total=len(X)*2, desc=f"Preprocessing {dataset_name}") as pbar_pre:
        stemmed = Parallel(n_jobs=-1)(delayed(preprocess)(text, 'stem') for text in X)
        lemmatized = Parallel(n_jobs=-1)(delayed(preprocess)(text, 'lemma') for text in X)
        X_processed = {'stem': stemmed, 'lemma': lemmatized}
        pbar_pre.update(len(X)*2)

    models = [
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        GradientBoostingClassifier(random_state=42),
        MLPClassifier(max_iter=100, early_stopping=True, random_state=42),
        LinearSVC(max_iter=10000, random_state=42),
        MultinomialNB()
    ]
    
    results = []
    
    with tqdm(total=len(combinations)*len(models), desc=f"Training {dataset_name}") as pbar:
        for comb in combinations:
            curr_X = X_processed[comb['preproc']]
            X_train, X_test, y_train, y_test = train_test_split(
                curr_X, y, test_size=0.2, random_state=42
            )
            
            batch_models = [model for model in models 
                          if not (isinstance(model, MultinomialNB) and comb['lsa'])]
            
            batch_results = Parallel(n_jobs=-1, verbose=10)(
                delayed(process_combination)(
                    comb, model, X_train, y_train, X_test, y_test, dataset_name
                ) for model in batch_models
            )
            
            results.extend([res for res in batch_results if res is not None])
            pbar.update(len(batch_models))
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    data = load_data()
    results = []
    
    for dataset_name in ['ag_news', 'imdb']:
        X, y = data[dataset_name]
        df = evaluate_dataset(X, y, dataset_name)
        results.append(df)
    
    full_results = pd.concat(results)
    full_results.to_csv('text_classification_results.csv', index=False)
    print("\nTop configurations per dataset:")
    print(full_results.groupby('dataset').apply(lambda x: x.nlargest(3, 'f1_score')).to_string())
