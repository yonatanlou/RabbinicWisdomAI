import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from text_processing import load_and_process_data

EXPERIMENT_NAME = "init"
question_and_answers_df = load_and_process_data(EXPERIMENT_NAME)

# Load the model and tokenizer
# model_name = 'dicta-il/dictalm2.0-instruct'
model_name = 'unsloth/Phi-3-mini-4k-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# Step 1: Generate new answers
def generate_answer(question, target_length):
    input_text = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max(input_ids.shape[1], target_length) + 20,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7
    )

    generated_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_answer.split("Answer:")[1].strip()


question_and_answers_df['generated_answer'] = ''
question_and_answers_df = question_and_answers_df.head(10)

# Iterate through the DataFrame rows
for index, row in question_and_answers_df.iterrows():
    try:
        # Get the question and the length of the answer
        question = row['question']
        answer_length = len(row['answer'].split(" "))

        # Generate the answer
        generated_answer = generate_answer(question, answer_length)
        print(f"Question: {question}")
        print(f"{generated_answer=}")
        print(f"real answer: {row['answer']=}")
        # Assign the generated answer to the DataFrame
        question_and_answers_df.at[index, 'generated_answer'] = generated_answer

        # Optional: Print progress
        if index % 10 == 0:  # Print every 10 rows
            print(f"Processed {index} rows")

    except Exception as e:
        print(f"Error processing row {index}: {str(e)}")
        print(f"Question: {question}")
        print(f"Answer length: {answer_length}")
        # Optionally, you can continue with the next iteration or break the loop
        # continue  # Use this to skip to the next iteration
        # break  # Use this to stop the loop entirely

# Print completion message
print("Finished generating answers")


# Step 2: Match generated answers to most similar existing answers
def get_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()


# Get embeddings for all answers and generated answers
all_answers = question_and_answers_df['answer'].tolist() + question_and_answers_df['generated_answer'].tolist()
embeddings = [get_embedding(answer) for answer in tqdm(all_answers, desc="Getting embeddings")]

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)

# Find most similar answer for each generated answer
n_questions = len(question_and_answers_df)
most_similar = []

for i in range(n_questions):
    generated_idx = n_questions + i
    similarities = similarity_matrix[generated_idx, :n_questions]
    most_similar_idx = similarities.argmax()
    most_similar.append(most_similar_idx)

question_and_answers_df['most_similar_answer_index'] = most_similar
question_and_answers_df['most_similar_answer'] = question_and_answers_df['most_similar_answer_index'].apply(
    lambda x: question_and_answers_df.iloc[x]['answer'])

# Calculate accuracy
accuracy = (question_and_answers_df['most_similar_answer_index'] == question_and_answers_df.index).mean()
print(f"Accuracy: {accuracy:.2f}")

# Save results
question_and_answers_df.to_csv('results.csv', index=False)
