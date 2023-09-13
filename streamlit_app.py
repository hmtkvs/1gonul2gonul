import streamlit as st
import requests
import sqlite3
from bs4 import BeautifulSoup
import random
import time
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import openai
from transformers import pipeline

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

@st.cache_resource(show_spinner="Loading Pipeline...")
def load_pos_pipe():
    return pipeline("ner", model="wietsedv/xlm-roberta-base-ft-udpos28-tr")

# Add a refresh button
refresh_button = st.button('🔄')
# Reset the app to the initial state when the refresh button is clicked
if refresh_button:
    st.session_state.cache = {}
    st.session_state.highlighted_text = ""
    st.session_state.user_input = ""
    st.session_state.color_map = {}
    st.experimental_rerun()  # This will rerun the script, effectively clearing the text area


# SQLite database setup
conn = sqlite3.connect("term_cache.db")
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS term_cache (term TEXT PRIMARY KEY, definition TEXT);"""
)
conn.commit()

color_map = {}  # Map terms to their associated colors

def calculate_readable_color(r, g, b):
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 128 else "black"


def fetch_word_definition(word):
    # Check if the word is already in the database cache
    cursor.execute("SELECT definition FROM term_cache WHERE term = ?", (word,))
    db_result = cursor.fetchone()

    if db_result:
        return db_result[0]

    url = f"https://sozluk.adalet.gov.tr/{word}"
    response = requests.post(url, verify=False)

    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    ankat_div = soup.find("div", {"class": "ankat"})
    if ankat_div:
        terim_divs = ankat_div.find_all("div", {"class": "terim"})
        for terim in terim_divs:
            col_md_4 = terim.find("div", {"class": "col-md-4"})
            col_md_8 = terim.find("div", {"class": "col-md-8"})
            if col_md_4 and col_md_8:
                if col_md_4.text.strip().lower() == word.lower():
                    definition = col_md_8.text
                    # Add to database cache
                    cursor.execute(
                        "INSERT OR REPLACE INTO term_cache (term, definition) VALUES (?, ?)",
                        (word, definition),
                    )
                    conn.commit()
                    return definition
    return None


openai.api_key = st.secrets["openai_api_key"]


def generate_explanation(original_input, term_definitions):
    term_definitions_str = "\n".join(
        [f"{k}: {v}" for k, v in term_definitions.items()]
    )
    prompt = f"Original Input: {original_input}\nTerm Definitions: {term_definitions_str}"
    system_prompt = "Generate an explanation of this legal text in Turkish. Use term definitions instead of terms"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=200,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()

def reassemble_subtokens(ner_results):
    complete_words = []
    temp_word = ""
    for result in ner_results:
        current_word = result['word']
        # https://universaldependencies.org/u/pos/
        if result['entity'] in {'NOUN', 'PROPN', 'ADJ', 'VERB'}:  # You can expand this to include other POS tags
            if current_word.startswith('▁'):
                if temp_word:  # If temp_word is not empty, append it to complete_words
                    complete_words.append(temp_word)
                temp_word = current_word.strip('▁')
            else:
                temp_word += current_word
    if temp_word:  # Add the last temp_word if it's not empty
        complete_words.append(temp_word)
    return complete_words

def get_n_gram_definitions(user_input, n=2):
    pipe = load_pos_pipe()
    ner_results = pipe(user_input)
    candidate_words = list(set(reassemble_subtokens(ner_results)))
    n_gram_candidates = []
    for i in range(len(candidate_words)):
        for j in range(2, n + 1):
            n_gram = " ".join(candidate_words[i:i + j])
            # n_gram_candidates.append(n_gram)
            n_gram_candidates.append(candidate_words[i])
    return n_gram_candidates


st.title("🐙🐙🐙 Term Highlighter 🐙🐙🐙")

st.write(
    """
    <style>
        .highlight-box {
            border-radius: 5px;
            padding: 5px;
            margin: 2px;
            display: inline-block;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for Streamlit
if "cache" not in st.session_state:
    st.session_state.cache = {}
if 'highlighted_text' not in st.session_state:
    st.session_state.highlighted_text = ""
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "color_map" not in st.session_state:
    st.session_state.color_map = {}


user_input = st.text_area("Enter a paragraph:", value=st.session_state.user_input)
st.session_state.user_input = user_input
run_button = st.button("🔍 Fetch Terms 🔍")


if run_button and user_input:
    conn.commit()   
    st.session_state.cache = {}
    st.session_state.highlighted_text = ""

    with st.spinner("Fetching Terms..."):
        # Fetch definitions for n-grams and individual words.
        n_gram_words = get_n_gram_definitions(user_input)
        for word in n_gram_words:
            clean_word = word.strip(".,!?()[]{}\":;")
            definition = fetch_word_definition(clean_word)
            if definition:
                st.session_state.cache[clean_word] = definition  # Use session_state for cache
                if clean_word not in st.session_state.color_map:
                    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    st.session_state.color_map[clean_word] = "#{:02x}{:02x}{:02x}".format(r, g, b)
            time.sleep(0.5)

        # Highlight the original text based on the fetched definitions
        original_words = user_input.split()
        for original_word in original_words:
            clean_original_word = original_word.strip(".,!?()[]{}\":;")
            if clean_original_word in st.session_state.cache:
                color = st.session_state.color_map[clean_original_word]
                font_color = calculate_readable_color(
                    int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                )
                st.session_state.highlighted_text += f"<span class='highlight-box' style='background-color: {color}; color: {font_color};'>{original_word}</span> "
            else:
                st.session_state.highlighted_text += f"{original_word} "
            
    st.session_state.user_input = user_input

if st.session_state.highlighted_text:  # Display the highlighted text if available
    st.write(f"<br> {st.session_state.highlighted_text}", unsafe_allow_html=True)

if st.session_state.cache:
    st.sidebar.title("Term Definitions")
    for term, definition in st.session_state.cache.items():
        color = st.session_state.color_map.get(term, "#FFFFFF")
        font_color = calculate_readable_color(
            int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        )
        st.sidebar.write(
            f"<span style='background-color: {color}; color: {font_color};'>{term}</span>: {definition}",
            unsafe_allow_html=True,
        )

# Add another button for generating explanations
generate_explanation_button = st.button("💸 Generate Explanation 💸")

if generate_explanation_button and st.session_state.cache:
    with st.spinner("Generating Explanation..."):
        explanation = generate_explanation(user_input, st.session_state.cache)

    st.markdown("---")
    st.markdown("\n\n")
    st.markdown("## AI Generated Explanation")
    st.markdown(explanation, unsafe_allow_html=True)
