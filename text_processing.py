import docx
import re
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
        lines = paragraph.split('\n')
        if not lines:
            continue

        # Extract number and label from the first line
        first_line = lines[0]
        match = re.match(r'.*?(\d+\.*|\.*\d+)(.*)', first_line)
        if match:
            number, label = match.groups()
        else:
            print(first_line)
            number = "0"
            label = first_line




        # The content is everything after the first line
        content = '\n'.join(lines[1:]).strip()

        processed_paragraphs.append({
            'number': number.strip().replace(".",""),
            'label': label.strip(),
            'content': content,
            'content_chunked': content.strip().split('\n')
        })
    return processed_paragraphs

def print_stats(processed_paragraphs, df):
    questions_idxs = df["new_index"].to_list()
    extracted_answers_idsx = [int(l["number"]) for l in processed_paragraphs]
    missing_answers = set(questions_idxs) - set(extracted_answers_idsx)
    print("Missing answers: ", sorted(list(missing_answers)))
    print("Total paragraphs: ", len(processed_paragraphs))
    print("Total lines: ", sum(len(p['content']) for p in processed_paragraphs))
    print("Average lines per paragraph: ", sum(len(p['content_chunked']) for p in processed_paragraphs) / len(processed_paragraphs))
    print("Average words per line: ", sum(len(p['content'].split(" ")) for p in processed_paragraphs) / sum(len(p['content_chunked']) for p in processed_paragraphs))
    print("Total words: ", sum(len(p['content'].split(" ")) for p in processed_paragraphs))
# Main process
file_path = '/Users/yonatanlou/dev/RabbinicWisdomAI/files/Responsa Text.docx'
print("Reading file: ", file_path)
text = read_docx(file_path)

print("Splitting into paragraphs")
raw_paragraphs = split_into_paragraphs(text)

print("Processing paragraphs")
processed_paragraphs = process_paragraph(raw_paragraphs)

df = pd.read_csv("/Users/yonatanlou/dev/RabbinicWisdomAI/files/Listokin RA Project.csv")
df["new_index"] = range(2, len(df) + 2)
print_stats(processed_paragraphs, df)

# Split into paragraphs
# raw_paragraphs = split_into_paragraphs(text)
#
# # Process each paragraph
# processed_paragraphs = []
# for raw_paragraph in raw_paragraphs:
#     processed = process_paragraph(raw_paragraph)
#     if processed:
#         processed_paragraphs.append(processed)
# print(processed_paragraphs)
