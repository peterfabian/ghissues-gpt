import os
import streamlit as st
import openai
from ghissues_responder.functions import answer_question, load_embeddings

openai.api_key = os.getenv('OPENAI_KEY')


with st.spinner('Loading embeddings...'):
    # Load data into the dataframe.
    document_embeddings = load_embeddings("data/embeddings.csv.gz")
# Notify the reader that the data was successfully loaded.
st.success('Embedding data loaded! 🎉')



# Initialize session state
if "output_text" not in st.session_state:
    st.session_state['output_text'] = ""

# Page title
st.title("WooCommerce GH issues 🔮")

input_question = st.text_area(label='Enter question about GitHub issues:', value="", height=20)

summarize_values = st.checkbox("Summarize issues?", value=False)
st.button("Submit",
          on_click=answer_question,
          kwargs={"question": input_question, "context_embeddings": document_embeddings, "summarize_issues": summarize_values},
          )

output_text = st.text_area(label='Answer:', value=st.session_state['output_text'], height=250)