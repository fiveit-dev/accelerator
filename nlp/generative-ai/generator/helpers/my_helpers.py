def bprint(text):
    bold_text = f"\033[1m{text}\033[0m"
    print(bold_text)

def read_md(file_path):
    """Reads the content of a Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    return content

def read_docx(file_path):
    """Reads the content of a docx file."""
    from docx import Document
    doc = Document(file_path)
    text = ""

    for para in doc.paragraphs:
        text += para.text
        
    return text

def read_pdf(filepath):
    """Reads text from a PDF file."""
    from PyPDF2 import PdfReader
    text = ""
    reader = PdfReader(filepath)
    
    for page in reader.pages:
        text += page.extract_text()
        
    return text

def read_file(file_path):
    """Calls the appropiate read function for the input-file extension."""
    extension = file_path.split(".")[-1]
    
    if extension == "md":
        content = read_md(file_path) 
    elif extension == "pdf":
        content = read_pdf(file_path)
    elif extension == "docx":
        content = read_docx(file_path)
    else: 
        raise TypeError(f"File should be either md, pdf or docx. {extension} is not compatible.")

    return content

def chunk_text(full_text, max_words=100, overlap=0, separator=r"(?<=[.!?])\s+"):
    """
    Given a large string, returns a list of chunks based on a maximum word count, overlap, and separator pattern.
    
    Args:
    - full_text (str): The complete text to be chunked.
    - max_words (int): Maximum number of words per chunk.
    - overlap (int): Number of words to overlap between chunks.
    - separator (str): Regex pattern to split sentences based on punctuation.

    Returns:
    - List of text chunks.
    """
    import re
    
    # Split based on the separator, typically at sentence boundaries
    sentences = re.split(separator, full_text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        # Tokenize the sentence into words
        words = sentence.split()
        # Add to current chunk if within word limit
        if len(current_chunk) + len(words) <= max_words:
            current_chunk.extend(words)
        else:
            # Append the current chunk to chunks
            chunks.append(" ".join(current_chunk))
            # Start new chunk with an overlap only if overlap > 0
            current_chunk = (current_chunk[-overlap:] if overlap > 0 else []) + words

    # Append any remaining text as a final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_items_from_dict(data):
    # Choose a random top-level topic
    random_topic_key = random.choice(list(data.keys()))
    random_topic_data = data[random_topic_key]
    
    # Find a random example in the selected topic's data
    def find_random_value(d):
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively search in nested dictionaries
                result = find_random_value(value)
                if result:
                    return key, result
            elif isinstance(value, list):
                # Return a random item from the list
                return key, random.choice(value)
        return None

    # Get the example key and value
    example_key, example_value = find_random_value(random_topic_data)

    # Return using the dynamically found keys
    return {random_topic_key: random_topic_key, example_key: example_value}

def get_random_items_from_json(data):
    """
    input:
    {
    "key1": [str, str, str, ...],
    "key2": [list, list, list, ...],
    "key3": [dict, dict, dict, ...],
    ...
    }
    output:
    {
    "key1": random string,
    "key2": random list,
    random dict,
    ...
    }
    """
    import random
    
    output = {}
    for key, list_ in data.items():
        if isinstance(list_, list):
            value = random.choice(list_)
            if isinstance(value, dict):
                output.update(value)
            else:
                output[key] = value
        else:
            raise TypeError("Expected dictionary with lists as values.")
    return output