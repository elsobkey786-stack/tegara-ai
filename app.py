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

# قراءة مفاتيح التشغيل
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
    conn = sqlite3.connect(DB_PATH, timeout=20)

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

    except Exception:
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
# 5. قراءة قاعدة المعرفة
# ==========================================

def get_items():
    conn = connect()

    try:
        rows = conn.execute("""
            SELECT id, kind, title, question, answer, content
            FROM knowledge
            ORDER BY id DESC
        """).fetchall()

        return rows

    finally:
        conn.close()


# ==========================================
# 6. البحث في قاعدة المعرفة
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

        documents.extend(split_chunks(text))

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
            if similarities[index] >= 0.12:
                results.append(documents[index])

        return "\n\n".join(results)

    except Exception:
        return ""


# ==========================================
# 7. الاتصال بـ Gemini AI
# ==========================================

def ask_gemini(question, context):

    if not GEMINI_API_KEY:
        return (
            "مفتاح Gemini API غير مضبوط. "
            "يرجى إضافة GEMINI_API_KEY في إعدادات Secrets."
        )

    prompt = f"""
أنت Tegara AI، مساعد ذكي مخصص لطلاب كلية التجارة.

تعليمات مهمة:
1. أجب باللغة العربية الواضحة والبسيطة.
2. اجعل إجابتك منظمة وسهلة الفهم.
3. استخدم أمثلة عملية عند الحاجة.
4. استفد من المعلومات الموجودة في قاعدة المعرفة.
5. لا تخترع معلومات أو تنسب معلومة للمصادر دون دليل.
6. إذا لم تجد الإجابة في المعلومات المتاحة، وضح ذلك للطالب.
7. ساعد الطالب على فهم المواد والمفاهيم الدراسية.

المعلومات المتاحة من قاعدة المعرفة:
{context if context else "لا توجد معلومات مرتبطة بالسؤال في قاعدة المعرفة."}

سؤال الطالب:
{question}
"""

    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        # جلب الموديلات المتاحة لمفتاح API
available_models = []

for m in client.models.list():
    actions = m.supported_actions or []
    if "generateContent" in actions:
        available_models.append(m.name)

if not available_models:
    return "مفتاح Gemini لم يُرجع أي موديل يدعم توليد النصوص."

# اختيار موديل Flash متاح تلقائيًا
flash_models = [
    m for m in available_models
    if "flash" in m.lower()
    and "image" not in m.lower()
]

if flash_models:
    selected_model = flash_models[0]
else:
    selected_model = available_models[0]

response = client.models.generate_content(
    model=selected_model,
    contents=prompt
)

return response.text or "Gemini لم يُرجع إجابة."
        )

        answer = response.text

        if answer:
            return answer

        return "لم أتمكن من تكوين إجابة. حاول مرة أخرى."

    except Exception as e:
        error_text = str(e)
        status_code = getattr(e, "code", "غير معروف")

        if "API_KEY_INVALID" in error_text or "API key not valid" in error_text:
            return "مفتاح Gemini API غير صحيح أو غير مفعل."

        if "403" in error_text:
            return "لا يملك المفتاح صلاحية استخدام Gemini API."

        if "404" in error_text:
            return (
                "موديل Gemini غير متاح حاليًا. "
                "تحقق من اسم الموديل المتاح لحسابك."
            )

        if "429" in error_text:
            return (
                "وصلت إلى حد الاستخدام المسموح به في Gemini. "
                "حاول مرة أخرى لاحقًا."
            )

        if "503" in error_text:
            return (
                "خدمة Gemini مشغولة حاليًا. "
                "حاول مرة أخرى بعد قليل."
            )

        return f"""
حصل خطأ أثناء الاتصال بـ Gemini

كود الخطأ: {status_code}

تفاصيل الخطأ:
{error_text}
"""


# ==========================================
# 8. إضافة سؤال وإجابة
# ==========================================

def add_question(question, answer):
    conn = connect()

    try:
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
        return True

    finally:
        conn.close()


# ==========================================
# 9. حفظ سؤال وإجابة من لوحة المسؤول
# ==========================================

def save_qa_callback():

    question = st.session_state.get(
        "qa_question", ""
    ).strip()

    answer = st.session_state.get(
        "qa_answer", ""
    ).strip()

    if not question or not answer:
        st.session_state["qa_notice"] = (
            "من فضلك اكتب السؤال والإجابة أولًا."
        )

        st.session_state["qa_notice_type"] = "warning"
        return

    try:
        add_question(question, answer)

        st.session_state["qa_question"] = ""
        st.session_state["qa_answer"] = ""

        st.session_state["qa_notice"] = (
            "تم حفظ السؤال والإجابة بنجاح."
        )

        st.session_state["qa_notice_type"] = "success"

    except Exception as e:
        st.session_state["qa_notice"] = (
            f"لم يتم الحفظ. الخطأ: {e}"
        )

        st.session_state["qa_notice_type"] = "error"


# ==========================================
# 10. إضافة ملف إلى قاعدة المعرفة
# ==========================================

def add_file(title, content):
    conn = connect()

    try:
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

    finally:
        conn.close()


# ==========================================
# 11. حذف عنصر من قاعدة المعرفة
# ==========================================

def delete_item(item_id):
    conn = connect()

    try:
        conn.execute(
            "DELETE FROM knowledge WHERE id = ?",
            (item_id,)
        )

        conn.commit()

    finally:
        conn.close()


# ==========================================
# 12. واجهة التطبيق
# ==========================================

st.title("🎓 Tegara AI")

st.caption(
    "مساعدك الذكي للتعلم والدراسة في كلية التجارة"
)

page = st.sidebar.radio(
    "القائمة الرئيسية",
    [
        "💬 المساعد الذكي",
        "🔐 لوحة المسؤول"
    ]
)


# ==========================================
# 13. صفحة الطالب
# ==========================================

if page == "💬 المساعد الذكي":

    st.header("اسأل Tegara AI")

    st.write(
        "اكتب سؤالك الدراسي وسيحاول المساعد "
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

    question = st.chat_input(
        "اكتب سؤالك هنا..."
    )

    if question:

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("جاري تجهيز الإجابة..."):

                context = retrieve_context(question)
                answer = ask_gemini(question, context)

                st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })


# ==========================================
# 14. لوحة المسؤول
# ==========================================

elif page == "🔐 لوحة المسؤول":

    st.header("🔐 لوحة إدارة Tegara AI")

    if not ADMIN_PASSWORD:
        st.error(
            "لوحة المسؤول غير مفعلة. "
            "يرجى ضبط TEGARA_ADMIN_PASSWORD في Secrets."
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

        if "qa_notice" in st.session_state:

            notice = st.session_state["qa_notice"]

            notice_type = st.session_state.get(
                "qa_notice_type", "success"
            )

            if notice_type == "success":
                st.success(notice)

            elif notice_type == "warning":
                st.warning(notice)

            else:
                st.error(notice)

        st.text_area(
            "السؤال",
            key="qa_question",
            placeholder="اكتب السؤال هنا..."
        )

        st.text_area(
            "الإجابة",
            key="qa_answer",
            placeholder="اكتب الإجابة هنا..."
        )

        st.button(
            "💾 حفظ السؤال والإجابة",
            type="primary",
            on_click=save_qa_callback
        )

    # --------------------------------------
    # رفع الملفات
    # --------------------------------------

    with tab2:

        st.subheader("رفع ملف إلى قاعدة المعرفة")

        uploaded_file = st.file_uploader(
            "اختر ملف PDF أو Word",
            type=["pdf", "docx"],
            key="knowledge_file"
        )

        if uploaded_file:

            st.write(
                "اسم الملف:",
                uploaded_file.name
            )

            if st.button("رفع وحفظ الملف"):

                with st.spinner("جاري استخراج النص..."):

                    extracted_text = extract_file(
                        uploaded_file
                    )

                    if extracted_text:

                        try:
                            add_file(
                                uploaded_file.name,
                                extracted_text
                            )

                            st.success(
                                "تم رفع الملف وحفظ محتواه بنجاح."
                            )

                        except Exception as e:
                            st.error(
                                f"لم يتم حفظ الملف: {e}"
                            )

                    else:
                        st.error(
                            "لم يتم العثور على نص قابل للاستخراج. "
                            "تأكد أن الملف يحتوي على نص وليس صورًا فقط."
                        )

    # --------------------------------------
    # إدارة قاعدة المعرفة
    # --------------------------------------

    with tab3:

        st.subheader("إدارة قاعدة المعرفة")

        items = get_items()

        if not items:

            st.info(
                "لا توجد بيانات محفوظة حتى الآن."
            )

        else:

            st.write(
                f"عدد العناصر المحفوظة: {len(items)}"
            )

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
                        f"🗑️ حذف العنصر رقم {item_id}",
                        key=f"delete_{item_id}"
                    ):

                        try:
                            delete_item(item_id)

                            st.success("تم حذف العنصر.")
                            st.rerun()

                        except Exception as e:
                            st.error(
                                f"لم يتم الحذف: {e}"
                            )
