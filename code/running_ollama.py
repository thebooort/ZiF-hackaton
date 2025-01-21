from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import pandas as pd




# Sample DataFrame

import pandas as pd
df = pd.read_excel('../data/data.xlsx')

# Define the prompt template to extract necessary details
template = """
Text: {text}

Questions:
1. Where is the study made? 'location country'
2. What type of crop is being studied? 'Crop name'
3. What type of ecosystem is involved? 'Ecosystem type'
4. What type of agriculture is used in the study?  'Agriculture type'
5. Where region is the study made?  'Region name'

answer template you should follow:
1. 'Country' 
2. 'Crop'
3. 'Ecosystem'
4. 'Agriculture'
5. 'Region'

Answer:go directly to answers, 1 by 1. Give me short answers or say not stated. preserve the format.
"""

# Create the prompt using the template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the Ollama model
model = OllamaLLM(model="llama3.1")

# Chain the prompt with the model
chain = prompt | model

# Function to ask questions based on text
def ask_ollama_about_study(text, question=None):
    if question:
        response = chain.invoke({"text": text, "question": question})
    else:
        response = chain.invoke({"text": text})
    return response


# Lists to collect answers for each column
countries = []
crops = []
ecosystems = []
agricultures = []
regions = []


# Iterate through each row in the DataFrame and ask questions
for index, row in df.iterrows():
    study_text = str(row['name'])+str(row['abstract'])
    
    # Print the study description for debugging
    print(f"Study {row['pdf']} \n")
    
    # Ask Ollama the questions
    response = ask_ollama_about_study(study_text)
    answers = response.split('\n')
    
    country = answers[2].strip().split('.')[1] if len(answers) > 1 else None
    crop = answers[3].strip().split('.')[1] if len(answers) > 2 else None
    ecosystem = answers[4].strip().split('.')[1] if len(answers) > 3 else None
    agriculture = answers[5].strip().split('.')[1] if len(answers) > 4 else None
    region = answers[6].strip().split('.')[1] if len(answers) > 5 else None

    # Output the extracted information
    print("Country:", country)
    print("Crop:", crop)
    print("Ecosystem:", ecosystem)
    print("Agriculture:", agriculture)
    print("Region:", region)
    
    # Append the answers to the lists
    countries.append(country)
    crops.append(crop)
    ecosystems.append(ecosystem)
    agricultures.append(agriculture)
    regions.append(region)

# Add the results to the DataFrame
df['country'] = countries
df['crop'] = crops
df['ecosystem'] = ecosystems
df['agriculture'] = agricultures
df['region'] = regions

df.to_excel('../data/data_with_answers.xlsx', index=False)


q1, q2, q3, q4, q5,q6 = [], [], [], [], [],[]

recommendations = ["what is the specific crop studied?",
"what are the plant-insect relationships observed in the study?",
"what are the best ways to protect crop based on this study?",
"what beneficial insect-plant interactions are observed in this study?",
"what pest are dangerous?",
"which natural practices are suggested in the paper?"]

template2 = """
Text: {text}

Questions:{question}

Answer:go directly to answers. Give me good answer in a understable way or say not stated.
"""
# Chain the prompt with the model

# Create the prompt using the template
prompt2 = ChatPromptTemplate.from_template(template2)
chain2 = prompt2 | model
def ask_ollama_about_study2(text, question=None):
    if question:
        response = chain2.invoke({"text": text, "question": question})
    else:
        response = chain2.invoke({"text": text})
    return response


for index, row in df.iterrows():
    study_text = str(row['name'])+str(row['abstract'])+str(row['results'])
    
    # Print the study description for debugging
    print(f"Study {row['pdf']} \n")
    answers = []
    for rec in recommendations:
        # Ask Ollama the questions
        response = ask_ollama_about_study2(study_text, rec)
        answers.append(response)
        print(answers)
    q1.append(answers[0])
    q2.append(answers[1])
    q3.append(answers[2])
    q4.append(answers[3])
    q5.append(answers[4])
    q6.append(answers[5])

# Add the results to the DataFrame
df[recommendations[0]] = q1
df[recommendations[1]] = q2
df[recommendations[2]] = q3
df[recommendations[3]] = q4
df[recommendations[4]] = q5
df[recommendations[5]] = q6

df.to_excel('../data/data_with_answers.xlsx', index=False)

df.to_csv('../data/data_with_answers.csv', index=False)















