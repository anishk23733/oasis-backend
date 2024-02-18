from flask import Flask, request, jsonify
from rag import RAG
from dotenv import load_dotenv

from flask_cors import CORS, cross_origin
from pymongo import MongoClient

from langchain_together import Together
from langchain_iris import IRISVector
from langchain.embeddings import HuggingFaceEmbeddings

import json

# Load environment variables
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="msmarco-bert-base-dot-v5")

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

client = MongoClient()

username = 'SUPERUSER'
password = 'oasis' # Replace password with password you set 
hostname = 'localhost' 
port = '1972' 
namespace = 'USER'
CONNECTION_STRING = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"
COLLECTION_NAME = "vectordb"

db = IRISVector(
    embedding_function=embeddings,
    dimension=768,
    # dimension=384,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)

# Initialize your model here to use it across different requests
company = "NVIDIA Corporation"
model = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    # model="meta-llama/Llama-2-70b-chat-hf",
    temperature=0.7,
    max_tokens=1024,
    top_k=50
)
rag_instance = RAG(company, model)

def get_metrics(topic, phrase):
    global company
    pipeline = [
        {'$match': {'tags': {'$regex': phrase, '$options': 'i'}, 'metric': True, 'company': company, 'topic': topic}},
        {'$addFields': {'descriptionLength': {'$strLenCP': '$description'}}},
        {'$sort': {'descriptionLength': 1}},
        {'$limit': 2}
    ]
    # Execute the aggregation pipeline
    results = client.vectordb.company_key_data.aggregate(pipeline)
    # Print each document found
    return_res = []
    for document in results:
        return_res.append({'value': document['value'], 'description': document['description']})
    return return_res

def get_closest_feature(target_company, query, topic):
    docs_with_score = db.similarity_search_with_score(
    query, k=1, filter={'company': target_company, "metric": True, "topic": topic})

    doc, score = docs_with_score[0]
    res = doc.metadata

    return {
        "value": res['value'],
        "description": res['description']
    }

def generate_follow_up_questions(context):
    prompt = f"""
    <s>[INST] You are an agent speaking with a representative from {company}.
    Your goal is to assist the representative to help them better understand their sustainability practices.

    You provide follow up questions that the representative may ask.
    Use the data given to you to provide two follow up questions they could ask to better understand their data.

    You provide your output in JSON format, for example:
    ```
    [
        "What actions do you suggest to increase renewable energy use from 86% to 100%?",
        "How can we improve our diversity and inclusion efforts to balance the gender ratio currently at 80-20?"
    ]
    ```

    You only provide questions in JSON format as output. Do not provide your data in any other format.

    Given the following data, provide questions that the representative may ask:
    ```
    {context}
    ```
    [/INST]
    """

    result = None
    while not result:
        try:
            output = model(prompt)
            result = json.loads(output[output.index('['):output.index(']')+1])[:2]
        except:
            pass
    return result

def get_company_info_in_format():
    return {
        'Environmental': {
            'Energy Efficiency': get_metrics('E', 'efficiency'),
            'Renewable Energy': get_metrics('E', 'renewable'),
            'Carbon Emissions': get_metrics('E', 'emissions'),
            'Waste Management': get_metrics('E', 'waste'),
            'Water Management': get_metrics('E', 'water'),
        },
        'Social': {
            'Diversity': get_metrics('S', 'diversity'),
            'Inclusion': get_metrics('S', 'inclusion'),
            'Gender Diversity': get_metrics('S', 'gender'),
            'Education': get_metrics('S', 'education'),
        },
        'Governance': {
            'Compliance': get_metrics('G', 'compliance'),
            'Supply Chain': get_metrics('G', 'supply'),
        }
    }

@app.route('/chat', methods=['POST'])
@cross_origin()
def chat_with_bot():
    data = request.json

    user_prompt = data.get('prompt')

    if not user_prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Get response from RAG instance
    response = rag_instance.get_response(user_prompt)

    # Return the chatbot's response
    return jsonify({"response": response})


@app.route('/reset', methods=['POST'])
@cross_origin()
def reset_conversation():
    global company, rag_instance
    data = request.json

    # Reset conversation history to start fresh
    local_company = data.get('company')
    if local_company != company:
        company = local_company
        rag_instance = RAG(company, model)

    rag_instance.conversation_history = ""
    return jsonify({"message": "Conversation history reset"})


@app.route('/compare', methods=['POST'])
@cross_origin()
def compare_with():
    companies = client.vectordb.company_key_data.distinct('company')
    if company in companies:
        companies.remove(company)

    return jsonify({"companies": companies})


@app.route('/company_info', methods=['POST'])
@cross_origin()
def company_info():
    # Reset conversation history to start fresh
    data = request.json
    company = data.get('company')

    data = {}
    data['metadata'] = {
        'companyName': company,
        'stockExchange': 'NASDAQ'
    }
    data['data'] = get_company_info_in_format()

    return jsonify(data)

@app.route('/follow_up_questions', methods=['POST'])
@cross_origin()
def follow_up_questions():
    # Reset conversation history to start fresh
    data = request.json
    
    return jsonify({
        'questions': generate_follow_up_questions(get_company_info_in_format())
    })

@app.route('/comparison_company_info', methods=['POST'])
@cross_origin()
def comparison_company_info():
    # Reset conversation history to start fresh
    data = request.json
    company = data.get('company')
    company_data = data.get('company_data')
    comparison_company = data.get('comparison_company')

    data = {
        'metadata': {
            'companyName': comparison_company,
            'stockExchange': 'NASDAQ'
        },
        'data': {}
    }

    if not company_data:
        return jsonify(data)

    for category in company_data['data']:
        data['data'][category] = {}
        for subcategory in company_data['data'][category]:
            data['data'][category][subcategory] = []
            values = company_data['data'][category][subcategory]
            for item in values:
                cf = get_closest_feature(comparison_company, item['description'], category[0].upper())
                data['data'][category][subcategory].append(cf)

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
