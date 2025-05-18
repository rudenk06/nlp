from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')

dataset = load_dataset('ag_news')
train_data = dataset['train']['text']

def preprocess(text):
    text = text.lower() 
    text = text.translate(str.maketrans('', '', string.punctuation))  
    tokens = word_tokenize(text)  
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tokens


sentences = [preprocess(text) for text in train_data]

model = Word2Vec(
    sentences=sentences,
    vector_size=300,  # Размерность векторов
    window=5,         # Размер окна контекста
    min_count=5,      # Игнорировать слова с частотой < 5
    sg=1,             # Использовать Skip-gram
    workers=4,        # Количество потоков
    epochs=10,        # Количество эпох
    negative=5,       # Негативное сэмплирование
    sample=1e-3       # Субсемплинг частых слов
)

test_words = [
    "president", "technology", "company", 
    "road", "election", "sky", 
    "music", "university", "climate", "movie"
]

for word in test_words:
    try:
        similar = model.wv.most_similar(word, topn=5)  
        print(f"Слова, близкие к '{word}':")
        for sim_word, score in similar:
            print(f"  {sim_word} ({score:.2f})")
        print("---")
    except KeyError:
        print(f"Слово '{word}' отсутствует в словаре модели.\n---")

    
