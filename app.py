import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import boto3
import zipfile
import uuid
import time
from decimal import Decimal
import json

# --- Configuration ---
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
DYNAMODB_TABLE_NAME = "LawAdvisorChats"
AWS_REGION = os.environ.get("AWS_REGION")

LOCAL_MODEL_PATH = '/tmp/model/'
LOCAL_DATA_PATH = '/tmp/data/'
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

# --- AWS Service Clients (with region specified) ---
s3 = boto3.client('s3', region_name=AWS_REGION)
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# --- Helper Functions ---
def download_assets_from_s3():
    if os.path.exists(os.path.join(LOCAL_MODEL_PATH, 'model.safetensors')):
        print("Files already exist in /tmp. Skipping download.")
        return
    
    try:
        s3_model_key = 'model/model.zip'
        local_model_zip_path = os.path.join(LOCAL_MODEL_PATH, 'model.zip')
        print(f"Downloading {s3_model_key} from S3...")
        s3.download_file(S3_BUCKET_NAME, s3_model_key, local_model_zip_path)
        print("Unzipping model...")
        with zipfile.ZipFile(local_model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(LOCAL_MODEL_PATH)
        os.remove(local_model_zip_path)
    except Exception as e:
        print(f"FATAL: Could not download model from S3. Error: {e}")
        raise e

    files_to_download = {
        'data/Embedding.npy': os.path.join(LOCAL_DATA_PATH, 'Embedding.npy'),
        'data/laws.xlsx': os.path.join(LOCAL_DATA_PATH, 'laws.xlsx')
    }
    for s3_key, local_path in files_to_download.items():
        try:
            print(f"Downloading {s3_key} from S3...")
            s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
        except Exception as e:
            print(f"FATAL: Could not download {s3_key} from S3. Error: {e}")
            raise e

def default_converter(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    if isinstance(o, (np.floating, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Decimal):
        return float(o)

# --- Load Assets ---
try:
    download_assets_from_s3()
    load_dotenv()

    client = OpenAI()
    openai_enabled = True

    print("Loading model and data from local storage...")
    model = SentenceTransformer(LOCAL_MODEL_PATH)
    law_embeddings = np.load(os.path.join(LOCAL_DATA_PATH, 'Embedding.npy'))
    df_laws = pd.read_excel(os.path.join(LOCAL_DATA_PATH, 'laws.xlsx'))
    LAW_TEXT_COLUMN = 'text'
    if LAW_TEXT_COLUMN not in df_laws.columns:
        raise ValueError(f"Column '{LAW_TEXT_COLUMN}' not found.")
    law_texts_list = df_laws[LAW_TEXT_COLUMN].tolist()
    print("Model and data loaded successfully!")
    model_loaded = True
except Exception as e:
    print(f"FATAL: Could not load initial assets. App will not function correctly. Error: {e}")
    model_loaded = False

# --- Flask App Setup ---
app = Flask(__name__)
application = app

# --- API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        timestamp = data.get('timestamp')
        feedback_value = data.get('feedback')

        if not all([conversation_id, timestamp, feedback_value]):
            return jsonify({"error": "Missing required feedback data."}), 400

        print(f"Received feedback: conv_id={conversation_id}, ts={timestamp}, feedback={feedback_value}")

        table.update_item(
            Key={
                'conversation_id': conversation_id,
                'timestamp': timestamp
            },
            UpdateExpression="SET feedback = :val",
            ExpressionAttributeValues={
                ':val': feedback_value
            }
        )
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error saving feedback to DynamoDB: {e}")
        return jsonify({"error": "Could not save feedback."}), 500


@app.route('/get_chat', methods=['POST'])
def get_chat():
    data = request.get_json()
    conversation_id = data.get('conversation_id')
    if not conversation_id:
        return jsonify({"error": "Conversation ID is required."}), 400
    try:
        response = table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('conversation_id').eq(conversation_id))
        items = sorted(response.get('Items', []), key=lambda x: x['timestamp'])
        for item in items:
            if 'sources' in item and isinstance(item['sources'], str):
                try:
                    item['sources'] = json.loads(item['sources'], parse_float=Decimal)
                except json.JSONDecodeError:
                    item['sources'] = []
        return jsonify({"history": items})
    except Exception as e:
        print(f"Error fetching chat from DynamoDB: {e}")
        return jsonify({"error": "Could not retrieve chat history."}), 500

@app.route('/search', methods=['POST'])
def search():
    if not model_loaded:
        return jsonify({"error": "The model is not ready."}), 503
    
    try:
        data = request.get_json()
        query = data.get('query')
        conversation_id = data.get('conversation_id')

        is_new_conversation = not conversation_id
        if is_new_conversation:
            conversation_id = str(uuid.uuid4())
            initial_message = {'conversation_id': conversation_id, 'timestamp': str(time.time()), 'role': 'assistant', 'content': 'أهلاً بك! كيف يمكنني مساعدتك اليوم؟'}
            table.put_item(Item=initial_message)

        if not query:
            return jsonify({"error": "Please provide a query."}), 400
        
        user_timestamp = str(time.time())
        table.put_item(Item={'conversation_id': conversation_id, 'timestamp': user_timestamp, 'role': 'user', 'content': query})
        
        response = table.query(KeyConditionExpression=boto3.dynamodb.conditions.Key('conversation_id').eq(conversation_id))
        db_history = sorted(response.get('Items', []), key=lambda x: x['timestamp'])

        history_for_gpt, existing_sources = [], []
        for item in db_history:
            if item.get('role') in ['user', 'assistant']:
                 history_for_gpt.append({'role': item['role'], 'content': item['content']})
            if 'sources' in item and isinstance(item['sources'], str):
                try:
                    existing_sources.extend(json.loads(item['sources']))
                except json.JSONDecodeError:
                    pass

        query_embedding = model.encode([query])
        similarities_for_new = cosine_similarity(query_embedding, law_embeddings)
        new_top_indices = np.argsort(similarities_for_new[0])[::-1][:5]
        newly_fetched_sources = [{'law_text': law_texts_list[int(index)], 'source_index': int(index)} for index in new_top_indices]
        combined_sources_map = {src['law_text']: src for src in existing_sources}
        for src in newly_fetched_sources:
            if src['law_text'] not in combined_sources_map: combined_sources_map[src['law_text']] = src
        all_candidate_sources = list(combined_sources_map.values())
        final_sources = []
        if all_candidate_sources:
            all_candidate_texts = [src['law_text'] for src in all_candidate_sources]
            all_candidate_embeddings = model.encode(all_candidate_texts)
            similarities_for_reranking = cosine_similarity(query_embedding, all_candidate_embeddings)
            reranked_indices = np.argsort(similarities_for_reranking[0])[::-1]
            final_sources = [all_candidate_sources[i] for i in reranked_indices[:7]]
        context_for_gpt = "".join([f"المصدر رقم [{i+1}]:\n{src['law_text']}\n\n" for i, src in enumerate(final_sources)])

        system_prompt = (
            "أنت مساعد قانوني خبير ومختص في القوانين السعودية. مهمتك هي الإجابة على سؤال المستخدم الأخير بدقة ووضوح، "
            "معتمداً **حصرياً** على نصوص المواد القانونية التي أزودك بها كمصادر وسياق المحادثة السابق. "
            "إذا لم تكن الإجابة موجودة بشكل واضح وصريح ضمن المصادر المقدمة، أجب بـ: "
            "'لا أجد إجابة واضحة في المصادر المتوفرة لدي بخصوص هذا السؤال.' "
            "لا تحاول أبداً استنتاج أو تخمين الإجابة. اذكر دائماً أرقام المصادر التي استخدمتها في إجابتك، مثال: [المصدر 1]."
        )
        messages = [{"role": "system", "content": system_prompt}] + history_for_gpt
        user_prompt_with_context = f"بناءً على المصادر التالية، أجب على السؤال.\n\n## المصادر:\n{context_for_gpt}\n\n## السؤال:\n{query}"
        messages.append({"role": "user", "content": user_prompt_with_context})
        response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.1)
        gpt_answer = response.choices[0].message.content

        assistant_timestamp = str(time.time())
        assistant_message = {
            'conversation_id': conversation_id,
            'timestamp': assistant_timestamp,
            'role': 'assistant',
            'content': gpt_answer,
            'sources': json.dumps(final_sources, default=default_converter)
        }
        table.put_item(Item=assistant_message)
        
        return jsonify({
            "conversation_id": conversation_id,
            "latest_message": {
                'role': 'assistant',
                'content': gpt_answer,
                'sources': final_sources,
                'timestamp': assistant_timestamp
            }
        })

    except Exception as e:
        print(f"CRITICAL ERROR in /search endpoint: {e}")
        return jsonify({"error": "A major unexpected error occurred on the server."}), 500

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5000)

