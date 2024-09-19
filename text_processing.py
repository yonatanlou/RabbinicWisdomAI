import os
import re

import docx
import pandas as pd


def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])


def split_into_paragraphs(text):
    # Split the text into paragraphs using numbers at the start of lines
    paragraphs = text.split("\n\n")
    return paragraphs


def process_paragraph(paragraphs):
    processed_paragraphs = []
    for paragraph in paragraphs:
        lines = paragraph.split("\n")
        if not lines:
            continue

        # Extract number and label from the first line
        first_line = lines[0]
        match = re.match(r".*?(\d+\.*|\.*\d+)(.*)", first_line)
        if match:
            number, label = match.groups()
        else:
            print(first_line)
            number = "0"
            label = first_line
        content = "\n".join(lines[1:]).strip()

        processed_paragraphs.append(
            {
                "index": number.strip().replace(".", ""),
                "label": label.strip(),
                "content": content,
                "content_chunked": content.split(" "),
            }
        )
    return processed_paragraphs


def print_stats(processed_paragraphs, df):
    questions_idxs = df["index"].to_list()
    extracted_answers_idsx = [int(l["index"]) for l in processed_paragraphs]
    missing_answers = set(questions_idxs) - set(extracted_answers_idsx)
    print("Missing answers: ", sorted(list(missing_answers)))
    print("Total paragraphs: ", len(processed_paragraphs))
    print("Total lines: ", sum(len(p["content"]) for p in processed_paragraphs))
    print(
        "Average lines per paragraph: ",
        sum(len(p["content_chunked"]) for p in processed_paragraphs)
        / len(processed_paragraphs),
    )
    print(
        "Average words per line: ",
        sum(len(p["content"].split(" ")) for p in processed_paragraphs)
        / sum(len(p["content_chunked"]) for p in processed_paragraphs),
    )
    print(
        "Total words: ", sum(len(p["content"].split(" ")) for p in processed_paragraphs)
    )


# Main process
def create_experiment_folder(experiment_name):
    """Create a folder for the experiment if it doesn't exist."""
    folder_path = os.path.join("experiments", experiment_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def load_and_process_data(experiment_name):
    """Load and process the data, saving intermediate results."""
    folder_path = create_experiment_folder(experiment_name)

    # Check if processed data already exists
    processed_file = os.path.join(folder_path, "processed_data.csv")
    if os.path.exists(processed_file):
        print("Loading pre-processed data...")
        return pd.read_csv(processed_file)

    # If not, process the data
    file_path = "files/Responsa Text.docx"
    csv_path = "/Users/yonatanlou/dev/RabbinicWisdomAI/files/Listokin RA Project.csv"

    print("Reading file: ", file_path)
    text = read_docx(file_path)

    print("Splitting into paragraphs")
    raw_paragraphs = split_into_paragraphs(text)

    print("Processing paragraphs")
    processed_paragraphs = process_paragraph(raw_paragraphs)

    df = pd.read_csv(csv_path)
    df["index"] = range(2, len(df) + 2)

    print_stats(processed_paragraphs, df)

    df = df.dropna(subset="השאילה")
    processed_paragraphs_df = pd.DataFrame(processed_paragraphs)
    processed_paragraphs_df["index"] = processed_paragraphs_df["index"].astype(int)

    question_and_answers_df = pd.merge(df, processed_paragraphs_df, on="index")
    question_and_answers_df_ = question_and_answers_df[
        ["index", "שם התשובה ומקורו", "השאילה", "label", "content"]
    ]
    question_and_answers_df_.rename(
        columns={"השאילה": "question", "content": "answer"}, inplace=True
    )

    # Save processed data
    question_and_answers_df_.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

    return question_and_answers_df_


def generate_ext_q_and_a(q_and_a, relevant_titles, answer_length):
    results = []
    tmp_q_and_a = q_and_a[q_and_a["title"].isin(relevant_titles)]
    for i, row in tmp_q_and_a.iterrows():
        paragraphs = row["paragraphs"]
        title = row["title"]

        # For each paragraph, access the context and questions
        for paragraph in paragraphs:
            context = paragraph["context"]
            qas = paragraph["qas"]

            # For each Q&A pair, extract the question and answer
            for qa in qas:
                tmp_result = {}
                question = qa["question"]

                answer = qa["answers"][0]
                start = answer["answer.start"]
                answer_text = answer["text"]

                before = (start - answer_length) if start > answer_length else start

                extracted_answer = context[before : start + answer_length]
                combined_answer = extracted_answer + ". לכן התשובה היא " + answer_text

                tmp_result.update(
                    {
                        "questions_q_a": question,
                        "answers_q_a": combined_answer,
                        "title": title,
                    }
                )
                results.append(tmp_result)
    return pd.DataFrame(results)
    # return questions_q_a, answers_q_a
