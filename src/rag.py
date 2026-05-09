import os
from google import genai
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

# إعداد مفتاح الـ API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def generate_rag_answer(query, top_k, raw_docs):
    if not top_k:
        return "No answer found"

    # 1. أخذ جميع المستندات الـ (top-k) وجمع نصوصها لتكوين السياق (context)
    texts = []
    for doc_id, _ in top_k:
        text = raw_docs.get(doc_id, "")
        if text:
            texts.append(text)
            
    context = "\n---\n".join(texts)
    
    if not context.strip():
        return "No context available to generate an answer."

    prompt = f"""
    Based on the following context, answer the query in a clear summary of 2 to 3 sentences maximum. 
    Do not copy the text directly, but paraphrase the answer clearly based only on the provided context.
    
    Context:
    {context}
    
    Query:
    {query}
    """
    
    # 3. إرسال السياق والسؤال إلى نموذج Gemini الجديد (gemini-3-flash-preview)
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer from LLM: {str(e)}"
