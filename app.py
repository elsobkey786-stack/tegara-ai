
import os
import re
import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from google import genai
from pypdf import PdfReader
from docx import Document

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 1. إعدادات التطبيق
# ==========================================

st.set_page_config(
    page_title="Tegara AI",
    page_icon="🎓",
    layout="wide"
)

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "tegara_ai.db"
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MODEL_NAME = "gemini-3.8-flash"

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

try:
    ADMIN_PASSWORD = st.secrets.get("TEGARA_ADMIN_PASSWORD", "")
except Exception:
    ADMIN_PASSWORD = os.getenv("TEGARA_ADMIN_PASSWORD", "")


# ==========================================
# 2. قاعدة البيانات
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


def get_items():
    conn = connect()

    try:
        return conn.execute("""
            SELECT id, kind, title, question, answer, content
            FROM knowledge
            ORDER BY id DESC
        """).fetchall()
    finally:
        conn.close()


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
    finally:
        conn.close()

    refresh_knowledge_cache()


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

    refresh_knowledge_cache()


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

    refresh_knowledge_cache()


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

            parts = [
                paragraph.text
                for paragraph in document.paragraphs
                if paragraph.text.strip()
            ]

            for table in document.tables:
                for row in table.rows:
                    parts.append(
                        " | ".join(cell.text for cell in row.cells)
                    )

            text = "\n".join(parts)

        else:
            return ""

        return text.strip()

    except Exception:
        return ""


# ==========================================
# 4. تجهيز النصوص
# ==========================================

def clean_text(text):
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_chunks(text, chunk_size=900, overlap=120):
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
# 5. البحث السريع في قاعدة المعرفة
# ==========================================

@st.cache_data(ttl=30)
def get_knowledge_data():
    return get_items()


@st.cache_resource
def build_search_index(documents):
    if not documents:
        return None, None

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=50000
    )

    matrix = vectorizer.fit_transform(documents)

    return vectorizer, matrix


def refresh_knowledge_cache():
    get_knowledge_data.clear()
    build_search_index.clear()


def retrieve_context(question):
    items = get_knowledge_data()

    if not items:
        return ""

    documents = []

    for item in items:
        item_id, kind, title, q, answer, content = item

        if kind == "qa":
            text = (
                f"عنوان السؤال: {title or ''}\n"
                f"السؤال: {q or ''}\n"
                f"الإجابة: {answer or ''}"
            )
        else:
            text = (
                f"اسم الملف: {title or ''}\n"
                f"المحتوى: {content or ''}"
            )

        documents.extend(split_chunks(text))

    if not documents:
        return ""

    try:
        vectorizer, document_matrix = build_search_index(
            tuple(documents)
        )

        if vectorizer is None:
            return ""

        question_vector = vectorizer.transform([question])

        similarities = cosine_similarity(
            question_vector,
            document_matrix
        ).flatten()

        best_indices = similarities.argsort()[::-1][:3]

        results = []

        for index in best_indices:
            if similarities[index] >= 0.10:
                results.append(documents[index])

        return "\n\n".join(results)

    except Exception:
        return ""


# ==========================================
# 6. الوقت والتاريخ في مصر
# ==========================================

def get_egypt_time():
    now = datetime.now(ZoneInfo("Africa/Cairo"))

    return (
        f"الوقت الآن في مصر: {now.strftime('%I:%M %p')}\n"
        f"التاريخ اليوم: {now.strftime('%Y-%m-%d')}\n"
        f"اليوم: {now.strftime('%A')}"
    )


def get_time_answer(question):
    q = clean_text(question).lower()

    time_words = [
        "الساعة كام",
        "الساعه كام",
        "الوقت كام",
        "كم الساعة",
        "كم الساعه",
        "ما الوقت",
        "الوقت الحالي",
        "الساعة الآن",
        "الساعه الان"
    ]

    date_words = [
        "تاريخ اليوم",
        "النهارده كام",
        "النهارده ايه",
        "النهاردة كام",
        "تاريخ النهارده",
        "ما تاريخ اليوم"
    ]

    if any(word in q for word in time_words):
        now = datetime.now(ZoneInfo("Africa/Cairo"))
        return f"الساعة الآن في مصر هي {now.strftime('%I:%M %p')}."

    if any(word in q for word in date_words):
        now = datetime.now(ZoneInfo("Africa/Cairo"))
        return (
            f"تاريخ اليوم هو {now.strftime('%Y-%m-%d')} "
            f"وهو يوم {now.strftime('%A')}."
        )

    return None


# ==========================================
# 7. الاتصال بـ Gemini
# ==========================================

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def ask_gemini(question, context):
    if not GEMINI_API_KEY:
        return (
            "مفتاح Gemini API غير موجود.\n"
            "تأكد من إضافة GEMINI_API_KEY في إعدادات Secrets."
        )

    time_answer = get_time_answer(question)

    if time_answer:
        return time_answer

    prompt = f"""
أنت Tegara AI، مساعد ذكي لطلاب كلية التجارة بجامعة المنصورة.

مهمتك:
- مساعدة الطلاب في الدراسة وفهم المواد.
- الإجابة باللغة العربية الواضحة والبسيطة.
- فهم السؤال حتى لو الطالب كتبه بصياغة مختلفة.
- تنظيم الإجابة واستخدام الأمثلة عند الحاجة.
- عدم اختراع معلومات تخص الجامعة أو الكلية أو اللوائح.
- إذا كان السؤال عن معلومة إدارية خاصة بالكلية وغير موجودة في السياق،
  وضح للطالب أن المعلومة غير متاحة في قاعدة البيانات.

بالنسبة للأسئلة التعليمية العامة:
يمكنك شرح المفاهيم العلمية والدراسية بشكل صحيح ومبسط،
حتى لو لم تكن موجودة في قاعدة المعرفة.

معلومات كلية التجارة المعتمدة الموجودة في قاعدة البيانات:
{context if context else "لا توجد معلومات مرتبطة بالسؤال في قاعدة البيانات."}

سؤال الطالب:
{question}

أجب مباشرة وباختصار مفيد.
"""

    try:
        client = get_gemini_client(GEMINI_API_KEY)

        response = client.interactions.create(
            model=MODEL_NAME,
            input=prompt
        )

        answer = getattr(response, "output_text", None)

        if answer and answer.strip():
            return answer.strip()

        return "لم يصلني رد نصي من Gemini. حاول مرة أخرى."

    except Exception as e:
        error = str(e)

        if "401" in error or "API_KEY_INVALID" in error:
            return "مفتاح Gemini غير صحيح أو غير صالح."

        if "403" in error:
            return "مفتاح Gemini لا يملك صلاحية استخدام الخدمة."

        if "404" in error:
            return (
                "الموديل المطلوب غير متاح لحسابك. "
                "راجع اسم الموديل وإمكانية استخدامه."
            )

        if "429" in error:
            return (
                "طلبات Gemini كثيرة حاليًا أو وصلت إلى حد الاستخدام. "
                "انتظر قليلًا ثم حاول مرة أخرى."
            )

        if "503" in error:
            return (
                "خدمة Gemini مشغولة حاليًا. "
                "حاول مرة أخرى بعد قليل."
            )

        return f"حدث خطأ أثناء الاتصال بـ Gemini:\n\n{error}"


# ==========================================
# 8. حفظ السؤال والإجابة من لوحة المسؤول
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
            f"لم يتم الحفظ: {e}"
        )
        st.session_state["qa_notice_type"] = "error"


# ==========================================
# 9. واجهة التطبيق
# ==========================================

connect()

st.title("🎓 Tegara AI")
st.caption(
    "مساعدك الذكي للتعلم والدراسة في كلية التجارة"
)

st.sidebar.title("Tegara AI")
st.sidebar.caption("كلية التجارة - جامعة المنصورة")

page = st.sidebar.radio(
    "القائمة الرئيسية",
    [
        "💬 المساعد الذكي",
        "🔐 لوحة المسؤول"
    ]
)


# ==========================================
# 10. صفحة الطالب
# ==========================================

if page == "💬 المساعد الذكي":

    st.header("اسأل Tegara AI")

    st.write(
        "اكتب سؤالك الدراسي وسيحاول المساعد "
        "تقديم إجابة واضحة ومفيدة."
    )

    now = datetime.now(ZoneInfo("Africa/Cairo"))

    st.caption(
        f"🕒 {now.strftime('%I:%M %p')} | "
        f"📅 {now.strftime('%Y-%m-%d')}"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 5])

    with col1:
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

            with st.spinner("جاري تجهيز الإجابة..."):

                try:
                    context = retrieve_context(question)
                    answer = ask_gemini(question, context)

                except Exception as e:
                    answer = f"حدث خطأ داخل التطبيق:\n\n{e}"

                st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })


# ==========================================
# 11. لوحة المسؤول
# ==========================================

elif page == "🔐 لوحة المسؤول":

    st.header("🔐 لوحة إدارة Tegara AI")

    if not ADMIN_PASSWORD:
        st.error(
            "لوحة المسؤول غير مفعلة. "
            "أضف TEGARA_ADMIN_PASSWORD في Secrets."
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

            if password == ADMIN_PASSWORD:
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

            st.write("اسم الملف:", uploaded_file.name)

            if st.button("رفع وحفظ الملف"):

                with st.spinner("جاري استخراج النص..."):

                    extracted_text = extract_file(uploaded_file)

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
                            st.error(f"لم يتم الحفظ: {e}")

                    else:
                        st.error(
                            "لم يتم العثور على نص قابل للاستخراج. "
                            "تأكد أن الملف يحتوي على نص وليس صورًا فقط."
                        )

    # --------------------------------------
    # إدارة المعرفة
    # --------------------------------------

    with tab3:

        st.subheader("إدارة قاعدة المعرفة")

        items = get_knowledge_data()

        if not items:
            st.info("لا توجد بيانات محفوظة حتى الآن.")

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
                        st.write((content or "")[:1500])

                    if st.button(
                        f"🗑️ حذف العنصر رقم {item_id}",
                        key=f"delete_{item_id}"
                    ):

                        try:
                            delete_item(item_id)
                            st.success("تم حذف العنصر.")
                            st.rerun()

                        except Exception as e:
                            st.error(f"لم يتم الحذف: {e}")
