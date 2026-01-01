
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.markdown("""
<style>
.footer { text-align: center; font-size: 20px; color: var(--muted); }
</style>
""", unsafe_allow_html=True)
# ----------------------
# Helper: Cached resources
# ----------------------
@st.cache_data(show_spinner=False)
def load_word_index(limit=None):
    # returns word_index and reverse_word_index
    word_index = imdb.get_word_index()
    if limit:
        # optionally keep only top-n words
        word_index = {k: v for k, v in word_index.items() if v <= limit}
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

@st.cache_resource(show_spinner=False)
def load_sentiment_model(path='simple_rnn_imdb.h5'):
    try:
        model = load_model(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise
    return model

# ----------------------
# Preprocess / Decode
# ----------------------
MAXLEN_DEFAULT = 500

def preprocess_text(text, word_index, maxlen=MAXLEN_DEFAULT):
    if not text:
        return np.zeros((1, maxlen), dtype=np.int32)
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen)
    return padded_review

def decode_review(encoded_review, reverse_word_index):
    # encoded_review expected as list of ints (without batch dim)
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 3])

# ----------------------
# Word importance (leave-one-out)
# ----------------------
def word_importance_leave_one_out(text, word_index, model, maxlen=MAXLEN_DEFAULT, limit_words=60):
    words = text.lower().split()
    if len(words) == 0:
        return []
    words = words[:limit_words]
    base_input = preprocess_text(' '.join(words), word_index, maxlen=maxlen)
    base_pred = float(model.predict(base_input, verbose=0)[0][0])
    importances = []
    for i in range(len(words)):
        # remove ith word
        temp_words = words[:i] + words[i+1:]
        temp_input = preprocess_text(' '.join(temp_words), word_index, maxlen=maxlen)
        temp_pred = float(model.predict(temp_input, verbose=0)[0][0])
        # positive delta => word contributes positively to sentiment
        delta = base_pred - temp_pred
        importances.append((words[i], delta))
    # sort by absolute contribution
    importances_sorted = sorted(importances, key=lambda x: abs(x[1]), reverse=True)
    return base_pred, importances_sorted

# ----------------------
# Small utilities
# ----------------------
SAMPLE_REVIEWS = [
    "What a fantastic movie! The story, acting and direction were top notch.",
    "I wasted two hours of my life. The plot was weak and the acting was terrible.",
    "A pleasant surprise — had fun the whole time, would watch again.",
    "Overhyped. It had flashes of good moments but mostly dragged on.",
]

# ----------------------
# App UI
# ----------------------
st.set_page_config(page_title='IMDB Sentiment — Polished', layout='wide', initial_sidebar_state='expanded')

# Load resources
word_index, reverse_word_index = load_word_index()
model = load_sentiment_model('simple_rnn_imdb.h5')

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", width=100)
    st.title('Configuration')
    maxlen = st.slider('Max sequence length', min_value=100, max_value=1000, value=500, step=50)
    threshold = st.slider('Decision threshold', 0.0, 1.0, 0.5, 0.01)
    
    st.markdown('### Quick Demos')
    for i, ex in enumerate(SAMPLE_REVIEWS):
        if st.button(f'Example {i+1}', key=f"ex_{i}"):
            st.session_state['example_text'] = ex
    
    st.divider()
    st.info("Adjust threshold to tune the model's sensitivity.")
   
# Main
st.title('IMDB Movie Review Sentiment Analyzer')
st.write('Enter a movie review below.')

col1, col2 = st.columns([2, 1])

with col1:
    # input area
    if 'example_text' in st.session_state:
        default_text = st.session_state.get('example_text', '')
    else:
        default_text = ''

    user_input = st.text_area('Movie Review', value=default_text, height=200)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button('Classify'):
            start_ts = time.time()
            padded = preprocess_text(user_input, word_index, maxlen=maxlen)
            try:
                pred = float(model.predict(padded, verbose=0)[0][0])
            except Exception as e:
                st.error(f'Prediction error: {e}')
                pred = 0.0
            sentiment = 'Positive' if pred > threshold else 'Negative'

            st.metric('Sentiment', sentiment, f'{pred:.4f}')

            # probability visuals
            st.subheader('Model confidence')
            st.write('Probability that the review is Positive')
            st.progress(pred)
            st.bar_chart({'Probability': [pred, 1-pred]}, use_container_width=True)

            # show decoded tokens and lengths
            decoded = decode_review(padded[0], reverse_word_index)
            with st.expander('Decoded tokens (first 200 tokens)'):
                st.write(decoded[:4000])

            # token-level importance (leave-one-out)
            if len(user_input.split()) == 0:
                st.info('Enter text to compute token importances.')
            else:
                st.subheader('Token-level contribution (leave-one-out, top 20)')
                base_pred, importances = word_importance_leave_one_out(user_input, word_index, model, maxlen=maxlen, limit_words=60)
                # prepare table
                top = importances[:20]
                rows = []
                for w, delta in top:
                    effect = '↑' if delta > 0 else '↓'
                    rows.append({'word': w, 'delta (base - without_word)': round(delta, 4), 'effect': effect})
                st.table(rows)

            # Downloadable report
            report = {
                'input': user_input,
                'prediction': float(pred),
                'sentiment': sentiment,
                'top_token_importances': rows,
                'maxlen_used': maxlen,
                'threshold': threshold,
            }
            st.download_button('Download JSON Report', data=json.dumps(report, indent=2), file_name='sentiment_report.json')

    with c2:
        if st.button('Clear'):
            st.session_state['example_text'] = ''
            st.rerun()
   

with col2:
  with st.container():
        st.markdown("""
        <div style=" padding: 20px; border-radius: 10px; border: 1px solid #eee;">
            <h4 style="margin-top:0;">💡 How to read this</h4>
                <ul style="font-size: 14px;">
                     <li> <b>Token-level importance helps show why the model predicted a label.</b></li>
                     <li><b> We show probability, visual bar and a downloadable JSON report for reproducibility.</b></li>
                     <li><b> Sidebar controls (maxlen, threshold) let you show different model behaviours interactively.</b></li>
                     <li><b> Use example buttons in the sidebar to quickly demo to interviewers.</b></li>
                </ul>    
                
      </div>
        """, unsafe_allow_html=True)
  st.divider()              

st.markdown('---')
st.subheader('📝 Text Diagnostics')
if user_input:
        tokens = user_input.split()
        st.write(f'**Token Count:** {len(tokens)}')
        st.write(tokens[:60])
else:
        st.write('No input yet.')

# Footer
st.markdown('---')
st.markdown('<div class="footer">Made with ❤️ by Raaj</div>', unsafe_allow_html=True)
