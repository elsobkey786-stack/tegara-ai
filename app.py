
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

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "tegara_ai.db"
UPLOAD_DIR = APP_DIR / "uploads"

UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Tegara AI",
    page_icon="🎓",
    layout="wide"
)

MODEL_NAME = "gemini-3.8-flash"
TIME_ZONE = ZoneInfo("Africa/Cairo")

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


def refresh_knowledge_cache():
    """تحديث الذاكرة المؤقتة بعد تغيير البيانات."""
    get_items.clear()
    build_search_index.clear()


# ==========================================
# 3. الوقت والتاريخ
# ==========================================

def get_current_time():
    now = datetime.now(TIME_ZONE)

    return (
        f"الساعة الآن {now.strftime('%I:%M %p')} "
        f"بتوقيت مصر."
    )


def get_current_date():
    now = datetime.now(TIME_ZONE)

    weekdays = [
        "الاثنين", "الثلاثاء", "الأربعاء",
        "الخميس", "الجمعة", "السبت", "الأحد"
    ]

    months = [
        "يناير", "فبراير", "مارس", "أبريل",
        "مايو", "يونيو", "يوليو", "أغسطس",
        "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"
    ]

    weekday = weekdays[now.weekday()]
    month = months[now.month - 1]

    return (
        f"اليوم {weekday}، "
        f"{now.day} {month} {now.year}."
    )


def get_time_or_date_answer(question):
    """إجابة فورية عن الوقت والتاريخ دون استدعاء Gemini."""

    q = normalize_text(question)

    time_words = [
        "الساعه كام", "الوقت كام", "كم الساعه",
        "ما الوقت", "الوقت الحالي", "الساعه الحاليه",
        "قول لي الساعه", "الساعة كام"
    ]

    date_words = [
        "النهارده كام", "تاريخ اليوم", "ما تاريخ اليوم",
        "النهارده ايه", "اليوم كام", "تاريخ النهارده",
        "ما هو تاريخ اليوم"
    ]

    asks_time = any(word in q for word in time_words)
    asks_date = any(word in q for word in date_words)

    if asks_time and asks_date:
        return get_current_date() + "\n" + get_current_time()

    if asks_time:
        return get_current_time()

    if asks_date:
        return get_current_date()

    return None


# ==========================================
# 4. تنظيف النصوص
# ==========================================

def normalize_text(text):
    text = str(text or "").strip().lower()

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text


def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
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
# 5. قراءة قاعدة المعرفة بسرعة
# ==========================================

@st.cache_data(ttl=30, show_spinner=False)
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
# 6. تجهيز محرك البحث مرة واحدة
# ==========================================

@st.cache_resource(show_spinner=False)
def build_search_index(items):

    documents = []

    for item in items:
        item_id, kind, title, question, answer, content = item

        if kind == "qa":
            text = (
                f"عنوان السؤال: {title or ''}. "
                f"السؤال: {question or ''}. "
                f"الإجابة: {answer or ''}."
            )

            documents.append(clean_text(text))

        else:
            file_chunks = split_chunks(content or "")

            for chunk in file_chunks:
                documents.append(
                    clean_text(f"{title or ''}. {chunk}")
                )

    documents = [doc for doc in documents if doc]

    if not documents:
        return None, None

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=30000,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(documents)

        return vectorizer, (matrix, documents)

    except Exception:
        return None, None


# ==========================================
# 7. البحث السريع في البيانات
# ==========================================

def retrieve_context(question):

    items = get_items()

    if not items:
        return ""

    vectorizer, index_data = build_search_index(items)

    if vectorizer is None or index_data is None:
        return ""

    matrix, documents = index_data

    try:
        question_vector = vectorizer.transform([question])

        similarities = cosine_similarity(
            question_vector,
            matrix
        ).flatten()

        best_indices = similarities.argsort()[::-1][:3]

        results = []

        for index in best_indices:
            score = similarities[index]

            if score >= 0.10:
                results.append(documents[index])

        return "\n\n".join(results)

    except Exception:
        return ""


# ==========================================
# 8. ردود فورية للتحيات والكلمات البسيطة
# ==========================================

def get_quick_answer(question):

    q = normalize_text(question)

    greetings = [
        "السلام عليكم",
        "سلام عليكم",
        "صباح الخير",
        "مساء الخير",
        "اهلا",
        "اهلا بيك",
        "مرحبا",
        "هاي",
        "hello",
        "hi"
    ]

    thanks = [
        "شكرا",
        "شكرا ليك",
        "متشكر",
        "تسلم",
        "جزاك الله خيرا"
    ]

    if q in greetings:
        return (
            "أهلًا بيك في Tegara AI 🎓\n\n"
            "أنا مساعدك الذكي لطلاب كلية التجارة. "
            "اسألني عن الدراسة أو المواد أو المعلومات "
            "الموجودة في قاعدة معرفة الكلية."
        )

    if q in thanks:
        return "العفو، تحت أمرك دائمًا! 🌷"

    if q in ["انت مين", "من انت", "اسمك ايه"]:
        return (
            "أنا Tegara AI، مساعد ذكي مخصص لمساعدة "
            "طلاب كلية التجارة."
        )

    return None


# ==========================================
# 9. الاتصال بـ Gemini
# ==========================================

@st.cache_resource
def get_gemini_client(api_key):
    return genai.Client(api_key=api_key)


def ask_gemini(question, context):

    if not GEMINI_API_KEY:
        yield "مفتاح Gemini API غير مضبوط في إعدادات Secrets."
        return

    prompt = f"""
أنت Tegara AI، مساعد ذكي متخصص في مساعدة طلاب كلية التجارة.

قواعد الإجابة:
1. أجب باللغة العربية الواضحة والبسيطة.
2. اجعل الإجابة مباشرة ومنظمة دون إطالة غير ضرورية.
3. لا تبدأ في تحليل طويل أو تفكير مطول.
4. استخدم المعلومات المتاحة من قاعدة المعرفة أولًا.
5. لا تخترع معلومات عن كلية التجارة أو الجامعة.
6. إذا لم تكن الإجابة موجودة في المعلومات المتاحة،
   وضح للطالب أن قاعدة المعرفة لا تحتوي على معلومة مؤكدة.
7. يمكنك شرح المفاهيم الدراسية العامة، لكن لا تنسبها
   إلى الكلية أو الجامعة دون مصدر.
8. إذا كان السؤال بسيطًا، أجب باختصار.
9. لا تكرر السؤال في بداية الإجابة.

المعلومات المسترجعة من قاعدة المعرفة:
{context[:3500] if context else "لا توجد معلومات مرتبطة بالسؤال في قاعدة المعرفة."}

سؤال الطالب:
{question}

اكتب الإجابة الآن بشكل واضح ومختصر.
"""

    try:
        client = get_gemini_client(GEMINI_API_KEY)

        stream = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            stream=True
        )

        received_text = False

        for event in stream:

            if getattr(event, "event_type", "") == "step.delta":

                delta = getattr(event, "delta", None)

                if getattr(delta, "type", None) == "text":

                    chunk = getattr(delta, "text", "")

                    if chunk:
                        received_text = True
                        yield chunk

        if not received_text:
            yield "لم يتمكن المساعد من تكوين إجابة. حاول مرة أخرى."

    except Exception as e:

        error_text = str(e)

        if "401" in error_text:
            yield "مفتاح Gemini API غير صحيح."

        elif "403" in error_text:
            yield "مفتاح API لا يملك صلاحية استخدام الخدمة."

        elif "404" in error_text:
            yield "الموديل غير متاح. راجع اسم الموديل وإعدادات API."

        elif "429" in error_text:
            yield "وصلنا إلى حد الاستخدام. حاول مرة أخرى بعد قليل."

        elif "503" in error_text:
            yield "خدمة Gemini مشغولة حاليًا. حاول بعد قليل."

        elif "timeout" in error_text.lower():
            yield "الاتصال استغرق وقتًا طويلًا. حاول مرة أخرى."

        else:
            yield f"حدث خطأ أثناء الاتصال: {error_text}"


# ==========================================
# 10. استخراج النص من الملفات
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
# 11. إضافة سؤال وإجابة
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
            datetime.now(TIME_ZONE).isoformat()
        ))

        conn.commit()

    finally:
        conn.close()

    refresh_knowledge_cache()


# ==========================================
# 12. حفظ سؤال وإجابة من لوحة المسؤول
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
# 13. إضافة ملف إلى قاعدة المعرفة
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
            datetime.now(TIME_ZONE).isoformat()
        ))

        conn.commit()

    finally:
        conn.close()

    refresh_knowledge_cache()


# ==========================================
# 14. حذف عنصر من قاعدة المعرفة
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

    refresh_knowledge_cache()


# ==========================================
# 15. واجهة التطبيق
# ==========================================

connect()

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
# 16. صفحة الطالب
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

            # أولًا: فحص الوقت والتاريخ
            answer = get_time_or_date_answer(question)

            # ثانيًا: فحص التحيات والأسئلة البسيطة
            if answer is None:
                answer = get_quick_answer(question)

            # ثالثًا: البحث في قاعدة المعرفة ثم Gemini
            if answer is not None:
                st.markdown(answer)

            else:
                context = retrieve_context(question)

                answer = st.write_stream(
                    ask_gemini(question, context)
                )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })


# ==========================================
# 17. لوحة المسؤول
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

    # تسجيل الدخول
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
                            st.error(f"لم يتم حفظ الملف: {e}")

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
