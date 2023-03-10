import os
import json

from ghissues_responder.config import *
import numpy as np
import pandas as pd
import tiktoken
import openai
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit as st


encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))
model = SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "url", "text", "n_tokens", "[0.1, 0.2, ... ]" up to the length of the embedding vectors.
    """
    
    df = pd.read_csv(fname, header=0)
    
    # embeddings = []
    # for idx, row in df.iterrows():
        # issue_map[idx] = (row['url'], row['text'])
        # embeddings.append(np.fromstring(row['embeddings'][1:-1], sep=", ", dtype=np.float32))

    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array).apply(np.float32)
    df.set_index('url', inplace=True, drop=False)

    return df


def order_document_sections_by_query_similarity(query: str, document_embeddings: pd.DataFrame) -> list[(float, str, str)]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    top_k = 10
    cos_scores = util.cos_sim(query_embedding, document_embeddings['embeddings'])[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop " + str(top_k) + " most similar issues:")

    relevant_sections = []

    for score, idx in zip(top_results[0], top_results[1]):
        int_idx = int(idx)
        print(document_embeddings.iloc(0)[int_idx]['url'], "(Score: {:.4f})".format(score))
        if score > 0.3:
            relevant_sections.append((score, document_embeddings.iloc(0)[int_idx]))
    
    return relevant_sections


# document_embeddings = load_embeddings("processed/embeddings.csv")
# print("Loaded embeddings for " + str(len(document_embeddings)) + " documents")

# order_document_sections_by_query_similarity("What are most important issues related to performance?", document_embeddings)


def summarize_issue(issue_url: str, context_embeddings: pd.DataFrame) -> str:
    issue_texts = context_embeddings.loc[issue_url]['text']

    if isinstance(issue_texts, str):
        issue_texts = [issue_texts]

    summary = ""
    for issue_text in issue_texts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced and helpful software engineer empathizing with a user who is having trouble with a product. "},
                {"role": "user", "content": "Please summarize a user issue for me. Be concise and pay attention to details. The issue description follows: " + issue_text},
            ],
            temperature=0,
        )
        summary += response['choices'][0]['message']['content'] + "\n\n"

    return summary


def construct_prompt(question: str, context_embeddings: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_urls = []
     
    for score, df_row in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        chosen_sections_len += df_row['n_tokens'] + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + df_row['text'].replace("\n", " "))
        chosen_sections_urls.append(df_row['url'])
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_urls))
    
    header = """You're an experienced engineering director assessing issues for an ecommerce platform. Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return (header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_urls)



# summary = summarize_issue("https://github.com/woocommerce/woocommerce/issues/29386", document_embeddings)
# print(summary)

# prompt = construct_prompt("What are most important issues related to rounding taxes?", document_embeddings)
# print(prompt)

def construct_prompt_sum(question: str, context_embeddings: pd.DataFrame) -> str:
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_urls = []

    for score, df_row in most_relevant_document_sections:
        issue_summary = summarize_issue(df_row['url'], context_embeddings)
        issue_sumamry_len = len(encoding.encode(issue_summary))

        chosen_sections_len += issue_sumamry_len + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break

        chosen_sections.append(SEPARATOR + issue_summary.replace("\n", " "))
        chosen_sections_urls.append(df_row['url'])

    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_urls))

    header = """You're an experienced engineering director assessing issues for an ecommerce platform. Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return (header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:", chosen_sections_urls)

# prompt = construct_prompt_sum("What are most important issues related to rounding taxes?", document_embeddings)
# print(prompt)

def answer_question(question: str, context_embeddings: pd.DataFrame, summarize_issues=False) -> str:
    if summarize_issues:
        with st.spinner("Summarizing issues..."):
            (prompt, urls) = construct_prompt_sum(question, context_embeddings)
        st.success("Done!")
    else:
        (prompt, urls) = construct_prompt(question, context_embeddings)

    try:
        with st.spinner("Answering question..."):
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=150,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
        st.success("Done!")
        out = response['choices'][0]['text'].strip()
        out += "\n\n" + "This answer is based on the following issues:\n" + "\n".join(urls)
    except Exception as e:
        out = "AI brainfart: " + str(e)
    finally:
        st.session_state['output_text'] = out

# answer = answer_question("What are most important issues related to rounding tax?", document_embeddings)
# print(answer)