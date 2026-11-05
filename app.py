
import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st
from google import genai
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. إعدادات التطبيق
# ==========================================

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "tegara_ai.db"
UPLOAD_DIR = APP_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Tegara AI",
    page_icon="🎓",
    layout="wide"
)

# قراءة مفاتيح التشغيل من Streamlit Secrets
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    ADMIN_PASSWORD = st.secrets.get("TEGARA_ADMIN_PASSWORD", "")
except Exception:
    ADMIN_PASSWORD = os.getenv("TEGARA_ADMIN_PASSWORD", "")


# ==========================================
# 2. الاتصال بقاعدة البيانات
# ==========================================

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            title TEXT,
            question TEXT,
            answer TEXT,
            content TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    return conn


# ==========================================
# 3. استخراج النص من الملفات
# ==========================================

def extract_file(uploaded_file):
    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        elif filename.endswith(".docx"):
            document = Document(uploaded_file)
            text = "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

            for table in document.tables:
                for row in table.rows:
                    text += "\n" + " | ".join(
                        cell.text for cell in row.cells
                    )

        else:
            return None

        return text.strip()

    except Exception as e:
        st.error(f"حصل خطأ أثناء قراءة الملف: {e}")
        return None


# ==========================================
# 4. تقسيم النصوص
# ==========================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_chunks(text, chunk_size=900, overlap=150):
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ==========================================
# 5. قراءة البيانات من قاعدة المعرفة
# ==========================================

def get_items():
    conn = connect()

    rows = conn.execute("""
        SELECT id, kind, title, question, answer, content
        FROM knowledge
        ORDER BY id DESC
    """).fetchall()

    conn.close()
    return rows


# ==========================================
# 6. البحث داخل قاعدة المعرفة
# ==========================================

def retrieve_context(question):
    items = get_items()

    if not items:
        return ""

    documents = []

    for item in items:
        item_id, kind, title, q, answer, content = item

        if kind == "qa":
            text = f"""
            السؤال: {q or ""}
            الإجابة: {answer or ""}
            """
        else:
            text = content or ""

        for chunk in split_chunks(text):
            documents.append(chunk)

    if not documents:
        return ""

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4)
        )

        matrix = vectorizer.fit_transform(
            documents + [question]
        )

        question_vector = matrix[-1]
        document_vectors = matrix[:-1]

        similarities = cosine_similarity(
            question_vector,
            document_vectors
        ).flatten()

        best_indices = similarities.argsort()[::-1][:5]

        results = []

        for index in best_indices:
            score = similarities[index]

            if score >= 0.12:
                results.append(documents[index])

        return "\n\n".join(results)

    except Exception:
        return ""


# ==========================================
# 7. الاتصال بـ Gemini
# ==========================================

def ask_gemini(question, context):
    if not GEMINI_API_KEY:
        return "مفتاح Gemini API غير مضبوط. يرجى مراجعة إعدادات الموقع."

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
أنت Tegara AI، مساعد ذكي مخصص لطلاب كلية التجارة.

تعليمات مهمة:
1. أجب باللغة العربية الواضحة والبسيطة.
2. اجعل إجابتك منظمة وسهلة الفهم.
3. اعتمد على المعلومات الموجودة في قاعدة المعرفة عند توفرها.
4. لا تخترع معلومات أو تنسب معلومة للمصادر دون دليل.
5. إذا لم تجد الإجابة في المعلومات المتاحة، وضح ذلك للطالب.
6. ساعد الطالب في فهم المواد والمفاهيم الدراسية.

المعلومات المتاحة من قاعدة المعرفة:
{context if context else "لا توجد معلومات مرتبطة بالسؤال في قاعدة المعرفة."}

سؤال الطالب:
{question}
"""

        response = client.models.generate_content(
           model="gemini-3.8-flash",
            contents=prompt
        )

        if response.text:
            return response.text

        return "لم أتمكن من تكوين إجابة. حاول مرة أخرى."

    except Exception as e:
        return f"حصل خطأ أثناء الاتصال بخدمة الذكاء الاصطناعي: {e}"


# ==========================================
# 8. إضافة سؤال وإجابة
# ==========================================

def add_question(question, answer):
    conn = connect()

    conn.execute("""
        INSERT INTO knowledge
        (kind, title, question, answer, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "qa",
        question,
        question,
        answer,
        "",
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# ==========================================
# 9. إضافة ملف إلى قاعدة المعرفة
# ==========================================

def add_file(title, content):
    conn = connect()

    conn.execute("""
        INSERT INTO knowledge
        (kind, title, question, answer, content, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "file",
        title,
        "",
        "",
        content,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


# ==========================================
# 10. حذف عنصر من قاعدة المعرفة
# ==========================================

def delete_item(item_id):
    conn = connect()

    conn.execute(
        "DELETE FROM knowledge WHERE id = ?",
        (item_id,)
    )

    conn.commit()
    conn.close()


# ==========================================
# 11. واجهة التطبيق
# ==========================================

st.title("🎓 Tegara AI")
st.caption("مساعدك الذكي للتعلم والدراسة في كلية التجارة")

page = st.sidebar.radio(
    "القائمة الرئيسية",
    [
        "💬 المساعد الذكي",
        "🔐 لوحة المسؤول"
    ]
)


# ==========================================
# 12. صفحة الطالب
# ==========================================

if page == "💬 المساعد الذكي":

    st.header("اسأل Tegara AI")

    st.write(
        "اكتب سؤالك الدراسي، وسيحاول المساعد "
        "تقديم إجابة واضحة ومفيدة."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("🗑️ مسح المحادثة"):
        st.session_state.chat_history = []
        st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("اكتب سؤالك هنا...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("جاري البحث وتجهيز الإجابة..."):

                context = retrieve_context(question)
                answer = ask_gemini(question, context)

                st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })


# ==========================================
# 13. صفحة المسؤول
# ==========================================

elif page == "🔐 لوحة المسؤول":

    st.header("🔐 لوحة إدارة Tegara AI")

    # مهم: لو كلمة المرور غير موجودة، نمنع فتح الإدارة
    if not ADMIN_PASSWORD:
        st.error(
            "لوحة المسؤول غير مفعلة. "
            "يرجى ضبط TEGARA_ADMIN_PASSWORD في إعدادات Secrets."
        )
        st.stop()

    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:

        password = st.text_input(
            "كلمة مرور المسؤول",
            type="password",
            placeholder="أدخل كلمة المرور"
        )

        if st.button("دخول", type="primary"):

            if password and password == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("كلمة المرور غير صحيحة.")

        st.stop()

    st.success("تم تسجيل دخول المسؤول بنجاح.")

    if st.button("تسجيل الخروج"):
        st.session_state.admin_authenticated = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs([
        "➕ إضافة سؤال وإجابة",
        "📁 رفع ملف",
        "📚 إدارة المعرفة"
    ])

    # --------------------------------------
    # إضافة سؤال وإجابة
    # --------------------------------------

    with tab1:

        st.subheader("إضافة سؤال وإجابة جديدة")

        with st.form("add_qa_form"):

            new_question = st.text_area(
                "السؤال"
            )

            new_answer = st.text_area(
                "الإجابة"
            )

            submitted = st.form_submit_button(
                "حفظ السؤال والإجابة"
            )

            if submitted:

                if new_question.strip() and new_answer.strip():

                    add_question(
                        new_question.strip(),
                        new_answer.strip()
                    )

                    st.success("تمت إضافة السؤال والإجابة بنجاح.")
                    st.rerun()

                else:
                    st.warning("من فضلك اكتب السؤال والإجابة.")

    # --------------------------------------
    # رفع الملفات
    # --------------------------------------

    with tab2:

        st.subheader("رفع ملف إلى قاعدة المعرفة")

        uploaded_file = st.file_uploader(
            "اختر ملف PDF أو Word",
            type=["pdf", "docx"]
        )

        if uploaded_file:

            st.write("اسم الملف:", uploaded_file.name)

            if st.button("رفع وحفظ الملف"):

                with st.spinner("جاري استخراج النص وحفظه..."):

                    extracted_text = extract_file(uploaded_file)

                    if extracted_text:

                        add_file(
                            uploaded_file.name,
                            extracted_text
                        )

                        st.success(
                            "تم حفظ الملف وإضافته إلى قاعدة المعرفة."
                        )

                    else:
                        st.error(
                            "لم يتم العثور على نص قابل للاستخراج في الملف."
                        )

    # --------------------------------------
    # إدارة قاعدة المعرفة
    # --------------------------------------

    with tab3:

        st.subheader("إدارة قاعدة المعرفة")

        items = get_items()

        if not items:
            st.info("لا توجد بيانات محفوظة حتى الآن.")

        else:

            st.write(f"عدد العناصر المحفوظة: {len(items)}")

            for item in items:

                item_id, kind, title, question, answer, content = item

                with st.expander(
                    f"#{item_id} - {title or 'عنصر بدون عنوان'}"
                ):

                    if kind == "qa":

                        st.markdown("**السؤال:**")
                        st.write(question)

                        st.markdown("**الإجابة:**")
                        st.write(answer)

                    else:

                        st.markdown("**محتوى الملف:**")
                        st.write(
                            (content or "")[:1500]
                        )

                    if st.button(
                        f"حذف العنصر رقم {item_id}",
                        key=f"delete_{item_id}"
                    ):

                        delete_item(item_id)

                        st.success("تم حذف العنصر.")
                        st.rerun()
