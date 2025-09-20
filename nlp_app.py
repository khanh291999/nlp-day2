import streamlit as st
import spacy
from spacy import displacy
import pandas as pd

# Load the SpaCy model (assuming 'en_core_web_sm' is downloaded)
nlp = spacy.load("en_core_web_sm")

# Streamlit app title
st.title("NLP Pipeline Demo")

# Input text area for the user
text = st.text_area("Enter a paragraph of text:", height=150)

# Process the text if input is provided
if text:
    doc = nlp(text)
    
    # 1. List of tokens after tokenization
    st.subheader("1. Tokens")
    tokens = [token.text for token in doc]
    st.write(tokens)
    
    # 2. Part-of-Speech (POS) tag for each token
    st.subheader("2. POS Tags")
    pos_data = [(token.text, token.pos_) for token in doc]
    pos_df = pd.DataFrame(pos_data, columns=["Token", "POS"])
    st.table(pos_df)
    
    # 3. Named Entities (NER) highlighted with labels
    st.subheader("3. Named Entities (Highlighted)")
    if doc.ents:
        html = displacy.render(doc, style="ent", jupyter=False)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.write("No named entities found.")