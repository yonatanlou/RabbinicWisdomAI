import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from text_processing import load_and_process_data

EXPERIMENT_NAME = "init"
question_and_answers_df = load_and_process_data(EXPERIMENT_NAME)

# Load the model and tokenizer
# model_name = 'dicta-il/dictalm2.0-instruct'
model_name = 'dicta-il/BEREL'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
def load_gen_ai_answers(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into individual answers
    answers = content.split("--end_of_answer--")
    print("Got {} answers".format(len(answers)))
    data = []
    for i, answer in enumerate(answers):
        data.append({'index': i+2, 'generated_answer': answer.strip()})

    # Create DataFrame and sort by index
    df = pd.DataFrame(data)
    df = df.sort_values('index').reset_index(drop=True)

    return df


def simple_chunking(text, chunk_size=128, overlap=0):
    """
    Split the input text into chunks of specified size with optional overlap.

    Args:
    text (str): The input text to be chunked.
    chunk_size (int): The desired size of each chunk (in words).
    overlap (int): The number of words to overlap between chunks.

    Returns:
    list: A list of text chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def get_embedding(answer: str, model, tokenizer, method="mean_last_hidden"):
    all_embeddings = []
    texts = simple_chunking(answer, chunk_size=100, overlap=10)

    for text in tqdm(texts, desc="Processing chunks", leave=False):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        if method == "mean_last_hidden":
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        elif method == "cls_token":
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        elif method == "pooler_output":
            embedding = outputs.pooler_output.cpu().numpy()
        elif method == "mean_all_layers":
            all_layers = outputs.hidden_states
            mean_all_layers = torch.stack(all_layers).mean(dim=0)
            embedding = mean_all_layers.mean(dim=1).cpu().numpy()
        else:
            raise ValueError(f"Unknown method: {method}")

        all_embeddings.append(embedding)

    return np.vstack(all_embeddings).mean(axis=0)  #

question_and_answers_df = question_and_answers_df.head(10)
# Step 1: Generate new answers
generated_answers = load_gen_ai_answers("files/genAI_answers.txt")
question_and_answers_df = pd.merge(question_and_answers_df, generated_answers, on='index', how='left')




# Get embeddings for all answers and generated answers
all_answers = question_and_answers_df['answer'].tolist() + question_and_answers_df['generated_answer'].tolist()
embeddings = []
for answer in tqdm(all_answers, desc="Getting embeddings"):
    embeddings.append(get_embedding(answer,model, tokenizer))

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)
n_questions = len(question_and_answers_df)
similarity_submatrix = similarity_matrix[n_questions:, :n_questions]

# Convert similarities to costs by subtracting from 1 (because the Hungarian algorithm finds minimum cost)
cost_matrix = 1 - similarity_submatrix

# Perform matching using the Hungarian algorithm
row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Store the matching results
question_and_answers_df['matched_answer_index'] = col_indices
question_and_answers_df['matched_answer'] = question_and_answers_df['matched_answer_index'].apply(
    lambda x: question_and_answers_df.iloc[x]['answer'])

question_and_answers_df.rename(columns={'index': 'origin_index'}, inplace=True)
question_and_answers_df["answer_length"] = question_and_answers_df["answer"].apply(lambda x: len(x.split()))

# Calculate accuracy
accuracy = (question_and_answers_df['matched_answer_index'] == question_and_answers_df.index).mean()
print(f"Accuracy: {accuracy:.2f}")

# Save results
question_and_answers_df.to_csv('results.csv', index=False)
#
# # Find most similar answer for each generated answer
# n_questions = len(question_and_answers_df)
# most_similar = []
#
# for i in range(n_questions):
#     generated_idx = n_questions + i
#     similarities = similarity_matrix[generated_idx, :n_questions]
#     most_similar_idx = similarities.argmax()
#     most_similar.append(most_similar_idx)
#
# question_and_answers_df['most_similar_answer_index'] = most_similar
# question_and_answers_df['most_similar_answer'] = question_and_answers_df['most_similar_answer_index'].apply(
#     lambda x: question_and_answers_df.iloc[x]['answer'])
# question_and_answers_df.rename(columns={'index': 'origin_index'}, inplace=True)
# question_and_answers_df["answer_length"] = question_and_answers_df["answer"].apply(lambda x: len(x.split()))
# # Calculate accuracy
# accuracy = (question_and_answers_df['most_similar_answer_index'] == question_and_answers_df.index).mean()
# print(f"Accuracy: {accuracy:.2f}")
#
# # Save results
# question_and_answers_df.to_csv('results.csv', index=False)
