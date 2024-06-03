import os
import pandas as pd
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import json


def model_fn(model_dir):
    """
    Load the model and tokenizer for inference.
    """
    print("Loading model.")
    model_path = os.path.join(model_dir, 'serverchat')
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForQuestionAnswering.from_pretrained(model_path)
    # model.eval()
    print("Model loaded.")

    # Load data
    data_path = os.path.join(model_dir, 'Extended_Data-2.xlsx')
    df = pd.read_excel(data_path, sheet_name="Sheet1")
    return {'model': model, 'tokenizer': tokenizer, 'dataframe': df}


def input_fn(request_body, content_type):
    """
    Deserialize and prepare the prediction input.
    """
    if content_type == 'application/json':
        # Ensure the body is decoded and loaded as JSON
        input_data = json.loads(request_body)
        return input_data['inputs']  # Adjust this according to how your JSON structure looks
    else:
        # Handle other content-types here or raise an exception
        raise ValueError("Unsupported content type: {}".format(content_type))



def extract_context(question, dataframe):
    keywords = question.split()
    question_clean = re.sub(r'[?.,!]', '', question)  # Clean question from common punctuation
    server_names = dataframe['Servers'].dropna().unique()
    server_name = next((server for server in server_names if server.lower() in question_clean.lower()), None)

    context = ""

    # Regex for environment questions, considering uppercased environment names
    match = re.search(r"all apps ruuning on (\w+) Environment", question_clean)
    if match:
        # Capture the environment name and convert to uppercase to match the DataFrame format
        environment_name = match.group(1).upper()  
        context_data = dataframe[dataframe['Environment'] == environment_name]
        if not context_data.empty:
            applications = ', '.join(context_data['Application_name'].unique())
            context = f"Applications in the {environment_name} environment are: {applications}."
        else:
            context = f"No applications found in the {environment_name} environment."
       
    # Specific handling for "Which users own X server?"
    if "which users own" in question.lower():
        try:
            # Extract server name after "which users own"
            server_name = question.lower().split("which users own")[1].strip()
            server_name = re.sub(r' server|\?', '', server_name).strip()  # Clean up string
            
            context_data = dataframe[dataframe['Servers'].str.contains(server_name, case=False, na=False)]
            if not context_data.empty:
                owners = ', '.join(set(context_data['Owner']))  # Remove duplicates and format nicely
                context = f"The users owning the server {server_name} are: {owners}."
            else:
                context = f"No users found for server {server_name}."
        except IndexError:
            context = "Error processing the question. Please check the format."
    
    return context


def answer_question(question, context, model, tokenizer):
    """
    Answer a question given a context using the loaded model and tokenizer.
    """
    # Encode question and context to fit model input constraints
    inputs = tokenizer.encode_plus(
        question, 
        context, 
        add_special_tokens=True, 
        max_length=512, 
        truncation=True, 
        return_tensors="pt"
    )
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert the tokens to the answer string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0, answer_start:answer_end]))
    return answer


def predict_fn(input_data, model_artifacts):
    """
    Generate predictions from the model.
    """
    model = model_artifacts['model']
    tokenizer = model_artifacts['tokenizer']
    dataframe = model_artifacts['dataframe']
    question = input_data

    print("Extracting context.")
    context = extract_context(question, dataframe)
    print("Answering question.")
    answer = answer_question(question, context, model, tokenizer)

    return answer



def output_fn(prediction_output, accept):
    """
    Serialize and prepare the prediction output.
    """
    if accept == "application/json":
        return json.dumps({'answer': prediction_output}), accept
    raise ValueError("Unsupported content type: {}".format(accept))
