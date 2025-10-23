import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate


from dotenv import load_dotenv


load_dotenv() # read .env

api_key = os.getenv("GOOGLE_API_KEY")
api_model = os.getenv("GEMINI_MODEL_NAME")


if "index" not in st.session_state:
    st.session_state.index = None
if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="AI Job Assistant", layout="centered")


st.markdown("""
<div style="text-align:center;">
    <h1>AI Job Assistant</h1>
    <p style="font-size:16px; color:gray;">
        Upload your resume or career documents and chat with an intelligent job assistant.<br>
        Get resume feedback, interview tips, and personalized career insights — all powered by AI.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()


with st.sidebar:
    st.header("Quick Guide")
    st.markdown("""
    1. Upload your resume or work experience documents (PDF, DOCX, MD).  
    2. Click “Build Vector Index” to create a searchable knowledge base.  
    3. Start chatting below — ask about resume improvement, career suggestions, or self-introduction writing.
                
    You can re-upload or rebuild your index anytime.
    
    
    """)

    st.info("Privacy Note: Your files are processed **only in your browser session** and never stored on any server.")

st.markdown("### Step 1: Upload Your Documents")

uploaded_files = st.file_uploader(
    "Upload your documents (PDF, DOCX, or MD)", 
    type=["pdf", "docx", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f'Uploaded {len(uploaded_files)} file(s) successfully')

st.divider()

st.markdown("### Step 2: Build Your Resume Knowledge Base")

if st.button("Build vector index"):
    if not uploaded_files:
        st.warning(' Please upload your files first')
    else:
        with st.status("Building vector index...", expanded=True) as status:
            st.write("Reading documents...")
            docs=[]

            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False,
                                                suffix=os.path.splitext(f.name)[1] #不设置这个的话，复制的文件将会没有扩展名
                                                ) as tmp_file:
                        file_content = f.read() # read upload file content
                        tmp_file.write(file_content) # write the file content into the temp file
                        tmp_path = tmp_file.name

                        ext = f.name.lower().split(".")[-1]
                        if ext == "pdf":
                            loader = PyPDFLoader(tmp_path)
                        elif ext == "docx":
                            loader = Docx2txtLoader(tmp_path)
                        elif ext in ["md"]:
                            loader = TextLoader(tmp_path, encoding="utf-8")
                        else:
                            st.warning(f"Not support this type files：{ext}")
                            continue

                        docs.extend(loader.load())

            st.write(f"Successfully loaded **{len(docs)}** document(s).")

            st.write("Splitting documents into chunks...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(docs)

            st.write("Generating text embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')

            st.write("Building FAISS vector index...")
            st.session_state.index = FAISS.from_documents(splits, embeddings)

            st.success(f" Vector index created with {len(splits)} text chunks.")
            status.update(label="Vector index successfully built!", state="complete", expanded=False)

st.divider()


st.markdown("### Step 3: Chat with Job Assistant")
st.caption(
    "You can ask questions like *“How can I improve my project descriptions?”* or *“Which projects should I highlight in my resume?”* "
    "You can also paste a job description (JD) — I'll analyze how well your resume matches it and suggest targeted improvements."
)


system_prompt = """

    你是智能求职助手，具备以下特征：
    - 专业、礼貌、逻辑清晰；
    - 熟悉简历优化、求职建议、项目经验总结；
    - 用英文回答时简洁流畅，语气自信自然；
    - 若用户用中文，你也可以用中文回应；
    - 回答时结合检索到的内容（用户的简历文本），不要凭空编造。

    任务目标：
    根据上传的简历文档和问题，帮用户分析优势、提出优化建议或生成改进版内容。

"""

user_input = st.chat_input("Ask your career or resume question here...")

if st.session_state.history:
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
            
if user_input:
    if not st.session_state.index:
        st.warning("Please upload your files and create the vector index first!")
    else:
        llm = ChatGoogleGenerativeAI(model=api_model,
                                     streaming=True
                                     )
        retriever = st.session_state.index.as_retriever(search_kwargs={"k": 3})
        related_docs = retriever.get_relevant_documents(user_input)
        context_text = "\n\n".join([doc.page_content for doc in related_docs])

        custom_prompt = ChatPromptTemplate.from_template(
        """
        {system_prompt}

        以下是检索到的相关内容：
        {context_text}

        历史对话：
        {chat_history}

        用户问题：
        {user_input}

        Please respond as a professional job assistant, focusing on resume writing, skill articulation, and career growth.
        """
        )
        chat_history_str = "\n".join(
            [f"User: {u}\nAssistant: {a}" for u, a in st.session_state.history]
        )

        filled_prompt = custom_prompt.format(
            system_prompt=system_prompt,
            context_text=context_text,
            chat_history=chat_history_str,
            user_input=user_input
            )

        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for chunk in llm.stream(filled_prompt):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)

        st.session_state.history.append((user_input, full_response))