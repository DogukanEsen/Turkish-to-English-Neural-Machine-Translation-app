from PyQt5.QtWidgets import QApplication, QMainWindow
from mainWin import Ui_MainWindow 
from transformers import pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import tensorflow as tf
import numpy as np


# HuggingFace
pipe1 = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-tr")
pipe2= pipeline("translation", model="Helsinki-NLP/opus-tatoeba-en-tr")
pipe3 = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en")
pipe4 = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-es")

# Oluşturulan Model
# Encoder
embedding_size = 100
num_encoder_words=21315
state_size = 512
dropout_rate=0.2
encoder_input = Input(shape=(None,), name='encoder_input')
encoder_embedding = Embedding(input_dim=num_encoder_words,
                              output_dim=embedding_size,
                              name='encoder_embedding')
encoder_lstm1 = LSTM(state_size,dropout=dropout_rate, name='encoder_lstm1', return_sequences=True)
encoder_lstm2 = LSTM(state_size,dropout=dropout_rate, name='encoder_lstm2', return_sequences=True)
encoder_lstm3 = LSTM(state_size,dropout=dropout_rate, name='encoder_lstm3', return_sequences=True)
encoder_lstm4 = LSTM(state_size,dropout=dropout_rate, name='encoder_lstm4', return_sequences=False)
def connect_encoder():
    net = encoder_input
    
    net = encoder_embedding(net)
    
    net = encoder_lstm1(net)
    net = encoder_lstm2(net)
    net = encoder_lstm3(net)
    net = encoder_lstm4(net)
    
    encoder_output = net
    
    return encoder_output
encoder_output = connect_encoder()
model_encoder = Model(inputs=[encoder_input], outputs=[encoder_output])

# Decoder
num_decoder_words=94058
decoder_initial_state = Input(shape=(state_size,), name='decoder_initial_state')
decoder_input = Input(shape=(None,), name='decoder_input')
decoder_embedding = Embedding(input_dim=num_decoder_words,
                              output_dim=embedding_size,
                              name='decoder_embedding')
decoder_lstm1 = LSTM(state_size,dropout=dropout_rate, name='decoder_lstm1', return_sequences=True)
decoder_lstm2 = LSTM(state_size,dropout=dropout_rate, name='decoder_lstm2', return_sequences=True)
# Cümle sequence olduğu için true olmalı
decoder_lstm3 = LSTM(state_size,dropout=dropout_rate, name='decoder_lstm3', return_sequences=True)
decoder_lstm4 = LSTM(state_size,dropout=dropout_rate, name='decoder_lstm4', return_sequences=True)
decoder_dense = Dense(num_decoder_words,
                      activation='linear',
                      name='decoder_output')
def connect_decoder(initial_state):
    net = decoder_input
    
    net = decoder_embedding(net)
    
    net = decoder_lstm1(net, initial_state=[initial_state, initial_state])
    net = decoder_lstm2(net, initial_state=[initial_state, initial_state])
    net = decoder_lstm3(net, initial_state=[initial_state, initial_state])
    net = decoder_lstm4(net, initial_state=[initial_state, initial_state])
    
    decoder_output = decoder_dense(net)
    
    return decoder_output
decoder_output = connect_decoder(initial_state=decoder_initial_state)

model_decoder = Model(inputs=[decoder_input, decoder_initial_state], outputs=[decoder_output])

# Load Weights
model_encoder.load_weights("model_encoder_50_epoch.keras")
model_decoder.load_weights("model_decoder_50_epoch.keras")

# Other Stuffs

mark_start= "baslangicccccccc "
mark_end= " endddddddddddddddd"
data_src = []
data_dest=[]
for line in open("Eng to tr.txt",encoding="UTF-8"):
    # tr ve ing cümleler tab ile ayrıldığı için tab'a göre split.
    en_text,tr_text=line.rstrip().split("\t")

    tr_text= mark_start + tr_text + mark_end
    data_src.append(en_text)
    data_dest.append(tr_text)
def tokenize_texts(texts, num_words=None):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)

    return tokenizer

def reverse_tokens(tokens):
    return [list(reversed(x)) for x in tokens]

def pad_tokens(tokens, maxlen, padding, truncating):
    return pad_sequences(tokens, maxlen=maxlen, padding=padding, truncating=truncating)

def calculate_max_tokens(num_tokens):
    return int(np.mean(num_tokens) + 2 * np.std(num_tokens))

def token_to_word(token, index_to_word):
    word=" " if token == 0 else index_to_word[token]
    return word
def tokens_to_string(tokens, index_to_word):
    words = [index_to_word[token] for token in tokens if token != 0]
    return " ".join(words)

def text_to_tokens(text, tokenizer, maxlen, padding, reverse=False):
    tokens = tokenizer.texts_to_sequences([text])
    tokens = np.array(tokens)

    if reverse:
        tokens = np.flip(tokens, axis=1)
        truncating = "pre"
    else:
        truncating = "post"

    return pad_sequences(tokens, maxlen=maxlen, padding=padding, truncating=truncating)

def tokenize_and_preprocess(texts, padding, reverse=False, num_words=None):
    tokenizer = tokenize_texts(texts, num_words=num_words)
    index_to_word = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
    
    tokens = tokenizer.texts_to_sequences(texts)

    if reverse:
        tokens = reverse_tokens(tokens)
        truncating = "pre"
    else:
        truncating = "post"

    num_tokens = [len(x) for x in tokens]
    max_tokens = calculate_max_tokens(num_tokens)

    tokens_padded = pad_tokens(tokens, maxlen=max_tokens, padding=padding, truncating=truncating)

    return {
        'tokenizer': tokenizer,
        'index_to_word': index_to_word,
        'tokens': tokens,
        'max_tokens': max_tokens,
        'tokens_padded': tokens_padded,
    }
tokenizer_src=tokenize_and_preprocess(texts=data_src,padding="pre",reverse=True,num_words=None)
tokenizer_dest=tokenize_and_preprocess(texts=data_dest,padding="post",reverse=False,num_words=None)
tokenizer_dest_word_index = tokenizer_dest["tokenizer"].word_index
tokenizer_src_word_index = tokenizer_src["tokenizer"].word_index
token_start = tokenizer_dest_word_index.get(mark_start.strip(), None)
token_end = tokenizer_dest_word_index.get(mark_end.strip(), None)

def translate_model(input_text):
    input_tokens =  text_to_tokens(input_text,tokenizer_src["tokenizer"],tokenizer_src["max_tokens"],"pre",False)
    initial_state = model_encoder.predict(input_tokens)
    max_tokens = tokenizer_dest["max_tokens"]
    
    decoder_input_data = np.zeros(shape=(1, max_tokens), dtype=np.int32)
    
    token_int = token_start
    output_text = ''
    count_tokens = 0
    
    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {'decoder_initial_state': initial_state, 'decoder_input': decoder_input_data}
        
        decoder_output = model_decoder.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]

        token_int = np.argmax(token_onehot)
        sampled_word = token_to_word(token_int,tokenizer_dest["index_to_word"])
        output_text += ' ' + sampled_word
        count_tokens += 1
    
    return output_text.rstrip("enddddddd")

#-----
class MyApplication(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.translateBtn.clicked.connect(self.translate_seq2seq)
        self.ui.translateBtn_2.clicked.connect(self.translate_hugging_face)
        self.ui.translateBtn_4.clicked.connect(self.translate_hugging_face_2)
        self.ui.translateBtn_5.clicked.connect(self.translate_hugging_face_3)
        self.ui.translateBtn_6.clicked.connect(self.translate_hugging_face_4)

    def translate_seq2seq(self):
        text = self.ui.engText.toPlainText()
        translated=translate_model(text)
        self.ui.trText.setPlainText(translated)
    def translate_hugging_face(self):
        text = self.ui.engText_2.toPlainText()
        translate = pipe1(text)[0]["translation_text"]
        self.ui.trText_2.setPlainText(translate)
    def translate_hugging_face_2(self):
        text=self.ui.engText_4.toPlainText()
        translate = pipe2(text)[0]["translation_text"]
        self.ui.trText_4.setPlainText(translate)
    def translate_hugging_face_3(self):
        text=self.ui.engText_5.toPlainText()
        translate = pipe3(text)[0]["translation_text"]
        self.ui.trText_5.setPlainText(translate)
    def translate_hugging_face_4(self):
        text=self.ui.engText_6.toPlainText()
        translate = pipe4(text)[0]["translation_text"]
        self.ui.trText_6.setPlainText(translate)


if __name__ == "__main__":
    app = QApplication([])
    window = MyApplication()
    window.show()
    app.exec_()