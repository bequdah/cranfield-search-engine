import os
import google.generativeai as genai

# إعداد مفتاح الـ API (يمكنك وضع المفتاح الخاص بك مباشرة بدل "YOUR_API_KEY_HERE" أو استخدامه كمتغير بيئة)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC1Zu2DMYQiAVxB0TGzaen3ISw7TMSKW0o")
genai.configure(api_key=GEMINI_API_KEY)

def generate_rag_answer(query, top_k, raw_docs):
    if not top_k:
        return "No answer found"

    # 1. أخذ أفضل 3 مستندات (top 3) وجمع نصوصها لتكوين السياق (context)
    texts = []
    for doc_id, _ in top_k[:3]:
        text = raw_docs.get(doc_id, "")
        if text:
            texts.append(text)
            
    context = "\n---\n".join(texts)
    
    if not context.strip():
        return "No context available to generate an answer."

    # 2. تجهيز الـ Prompt لإرساله للمودل للحصول على ملخص 2-3 جمل
    prompt = f"""
    Based on the following context, answer the query in a clear summary of 2 to 3 sentences maximum. 
    Do not copy the text directly, but paraphrase the answer clearly based only on the provided context.
    
    Context:
    {context}
    
    Query:
    {query}
    """
    
    # 3. إرسال السياق والسؤال إلى نموذج Gemini (استخدمنا gemini-2.5-flash لسرعته وكفاءته)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating answer from LLM: {str(e)}"