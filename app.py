import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dotenv import load_dotenv
import boto3 # AWS SDK
import zipfile # For unzipping the model

# --- Configuration ---
# !!! هام: غيّر هذا إلى اسم الـ bucket الفعلي الخاص بك !!!
S3_BUCKET_NAME = "saudi-law-advisor-assets"

# تحديد المسارات المحلية حيث سيتم تخزين الملفات
LOCAL_MODEL_PATH = './model/'
LOCAL_DATA_PATH = './data/'
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
os.makedirs(LOCAL_DATA_PATH, exist_ok=True)

# --- دالة لتحميل الملفات من S3 ---
def download_assets_from_s3():
    """Downloads model and data files from S3 if they don't exist locally."""
    # لا تقم بالتحميل إذا كنا نعمل محلياً ولدينا الملفات بالفعل لتسريع البدء
    if os.environ.get('FLASK_ENV') == 'development' and os.path.exists(os.path.join(LOCAL_MODEL_PATH, 'model.safetensors')):
        print("العمل في وضع التطوير، والملفات موجودة محلياً. تم تخطي التحميل من S3.")
        return

    s3 = boto3.client('s3')
    
    # --- تحميل النموذج ---
    s3_model_key = 'model/model.zip'
    local_model_zip_path = os.path.join(LOCAL_MODEL_PATH, 'model.zip')
    
    if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, 'model.safetensors')):
        print(f"جاري تحميل {s3_model_key} من S3...")
        try:
            s3.download_file(S3_BUCKET_NAME, s3_model_key, local_model_zip_path)
            print("جاري فك ضغط النموذج...")
            with zipfile.ZipFile(local_model_zip_path, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_MODEL_PATH)
            os.remove(local_model_zip_path)
            print("تم تحميل النموذج وفك ضغطه بنجاح.")
        except Exception as e:
            print(f"فشل تحميل النموذج من S3: {e}")
            return
    else:
        print("النموذج موجود محلياً.")

    # --- تحميل ملفات البيانات ---
    files_to_download = {
        'data/Embedding.npy': os.path.join(LOCAL_DATA_PATH, 'Embedding.npy'),
        'data/laws.xlsx': os.path.join(LOCAL_DATA_PATH, 'laws.xlsx')
    }

    for s3_key, local_path in files_to_download.items():
        if not os.path.exists(local_path):
            print(f"جاري تحميل {s3_key} من S3...")
            try:
                s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
                print(f"تم تحميل {s3_key} بنجاح.")
            except Exception as e:
                print(f"فشل تحميل {s3_key} من S3: {e}")
        else:
            print(f"{s3_key} موجود محلياً.")

# --- تحميل الملفات قبل بدء التطبيق ---
download_assets_from_s3()

# --- تحميل الإعدادات الأولية ---
load_dotenv()
try:
    client = OpenAI()
    openai_enabled = True
except Exception as e:
    print(f"تحذير: لم يتم إعداد OpenAI. السبب: {e}")
    openai_enabled = False

print("جاري تحميل النموذج والبيانات من التخزين المحلي...")
try:
    model = SentenceTransformer(LOCAL_MODEL_PATH)
    law_embeddings = np.load(os.path.join(LOCAL_DATA_PATH, 'Embedding.npy'))
    df_laws = pd.read_excel(os.path.join(LOCAL_DATA_PATH, 'laws.xlsx'))
    LAW_TEXT_COLUMN = 'text'
    if LAW_TEXT_COLUMN not in df_laws.columns:
        raise ValueError(f"العمود '{LAW_TEXT_COLUMN}' غير موجود.")
    law_texts_list = df_laws[LAW_TEXT_COLUMN].tolist()
    print("تم تحميل النموذج والبيانات بنجاح!")
    model_loaded = True
except Exception as e:
    print(f"حدث خطأ فادح أثناء تحميل النموذج أو البيانات: {e}")
    model_loaded = False

# --- إعداد تطبيق فلاسك ---
app = Flask(__name__)
application = app

# --- نقاط الوصول ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if not model_loaded:
        return jsonify({"error": "النموذج غير جاهز."}), 503

    data = request.get_json()
    query = data.get('query')
    history = data.get('history', [])
    existing_sources = data.get('sources', [])

    if not query:
        return jsonify({"error": "الرجاء إرسال استعلام."}), 400
    
    # --- المنطق الجديد: الدمج ثم إعادة الترتيب ---
    # 1. البحث عن مصادر جديدة للسؤال الحالي
    print(f"البحث عن مصادر جديدة للسؤال: '{query}'")
    query_embedding = model.encode([query])
    similarities_for_new = cosine_similarity(query_embedding, law_embeddings)
    new_top_indices = np.argsort(similarities_for_new[0])[::-1][:5] 

    newly_fetched_sources = [{'law_text': law_texts_list[int(index)], 'source_index': int(index)} for index in new_top_indices]

    # 2. دمج المصادر الجديدة مع القديمة بدون تكرار
    combined_sources_map = {source['law_text']: source for source in existing_sources}
    for source in newly_fetched_sources:
        if source['law_text'] not in combined_sources_map:
            combined_sources_map[source['law_text']] = source
    
    all_candidate_sources = list(combined_sources_map.values())
    all_candidate_texts = [source['law_text'] for source in all_candidate_sources]

    # 3. إعادة ترتيب كل المصادر المرشحة بناءً على صلتها بالسؤال الحالي
    if all_candidate_texts:
        print(f"إعادة ترتيب {len(all_candidate_texts)} مصدر مرشح...")
        all_candidate_embeddings = model.encode(all_candidate_texts)
        similarities_for_reranking = cosine_similarity(query_embedding, all_candidate_embeddings)
        
        reranked_indices = np.argsort(similarities_for_reranking[0])[::-1]
        
        # اختيار أفضل 7 مصادر بعد إعادة الترتيب
        final_sources = [all_candidate_sources[i] for i in reranked_indices[:7]]
    else:
        final_sources = []

    # 4. بناء السياق لـ GPT من أفضل المصادر النهائية
    context_for_gpt = ""
    for i, source in enumerate(final_sources):
        context_for_gpt += f"المصدر رقم [{i+1}]:\n{source['law_text']}\n\n"

    # --- التوليد (Generation) مع الموجه المحسن ---
    gpt_answer = "لم يتم تفعيل GPT-4o أو حدث خطأ."
    if openai_enabled:
        try:
            system_prompt = (
                "أنت مساعد قانوني خبير ومختص في القوانين السعودية. مهمتك هي الإجابة على سؤال المستخدم الأخير بدقة ووضوح، "
                "معتمداً **حصرياً** على نصوص المواد القانونية التي أزودك بها كمصادر وسياق المحادثة السابق. "
                "إذا لم تكن الإجابة موجودة بشكل واضح وصريح ضمن المصادر المقدمة، أجب بـ: "
                "'لا أجد إجابة واضحة في المصادر المتوفرة لدي بخصوص هذا السؤال.' "
                "لا تحاول أبداً استنتاج أو تخمين الإجابة. اذكر دائماً أرقام المصادر التي استخدمتها في إجابتك، مثال: [المصدر 1]."
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history)
            
            user_prompt_with_context = f"بناءً على المصادر التالية، أجب على السؤال.\n\n## المصادر:\n{context_for_gpt}\n\n## السؤال:\n{query}"
            messages.append({"role": "user", "content": user_prompt_with_context})

            response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.1)
            gpt_answer = response.choices[0].message.content
        except Exception as e:
            print(f"خطأ أثناء استدعاء OpenAI API: {e}")
            gpt_answer = f"حدث خطأ أثناء توليد الإجابة. التفاصيل: {e}"

    return jsonify({"gpt_answer": gpt_answer, "sources": final_sources})

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    application.run(debug=True, host='0.0.0.0', port=5000)

