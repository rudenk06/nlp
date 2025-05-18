import numpy as np
from tensorflow import keras
from keras import regularizers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Input
from tensorflow.keras.layers import Bidirectional, Attention, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from datasets import load_dataset


def build_and_train_model(X_train, y_train, X_val, y_val,
                          vocab_size, input_length,
                          num_classes,
                          lstm_units=64, embed_dim=100,
                          dropout_rate=0.3, learning_rate=0.005,
                          epochs=5, batch_size=64):

    final_activation = 'softmax'
    loss = 'sparse_categorical_crossentropy'
    output_units = num_classes

    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Input layer
    inp = Input(shape=(input_length,))
    x = Embedding(input_dim=vocab_size,
                  output_dim=embed_dim,
                  input_length=input_length)(inp)
    x = SpatialDropout1D(dropout_rate)(x)

    # Bidirectional LSTM
    x = Bidirectional(LSTM(
        lstm_units,
        dropout=dropout_rate,
        kernel_regularizer=regularizers.l2(0.001),
        recurrent_regularizer=regularizers.l2(0.001),
        return_sequences=True
    ))(x)

    # Attention mechanism
    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)

    # Output layer
    out = Dense(
        output_units,
        activation=final_activation,
        kernel_regularizer=regularizers.l2(0.001)
    )(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        loss=loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback]
    )
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=64)
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Test F1-score: {f1:.4f}")
    return f1


# Общие параметры
maxlen = 200
epochs = 5

# ----------------------------
# 1) AG News: 4‑классовая задача
# ----------------------------
ag = load_dataset("ag_news")
ag_text  = ag["train"]["text"] + ag["test"]["text"]
ag_label = np.concatenate([ag["train"]["label"], ag["test"]["label"]])  # метки 0–3

# Токенизация и паддинг
tokenizer_ag = Tokenizer()
tokenizer_ag.fit_on_texts(ag_text)
X_ag = tokenizer_ag.texts_to_sequences(ag_text)
X_ag = pad_sequences(X_ag, maxlen=maxlen)
vocab_ag = len(tokenizer_ag.word_index) + 1

# Train/validation/test split
X_ag_train, X_ag_test, y_ag_train, y_ag_test = train_test_split(
    X_ag, ag_label, test_size=0.2, random_state=42, stratify=ag_label)
X_ag_tr, X_ag_val, y_ag_tr, y_ag_val = train_test_split(
    X_ag_train, y_ag_train, test_size=0.1, random_state=42, stratify=y_ag_train)

print("\n=== AG News (4 classes) with Bi-LSTM + Attention ===")
model_ag = build_and_train_model(
    X_ag_tr, y_ag_tr, X_ag_val, y_ag_val,
    vocab_size=vocab_ag,
    input_length=maxlen,
    num_classes=4,
    epochs=epochs
)
evaluate_model(model_ag, X_ag_test, y_ag_test)