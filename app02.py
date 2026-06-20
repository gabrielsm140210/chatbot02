import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

st.set_page_config(page_title="PowerTech | Assistente Técnico CLP PT-3000", page_icon="⚙️", layout="centered")

COR_PRIMARIA = "#9BB33B"
COR_SECUNDARIA = "#F5AD03"

st.markdown(f"""
<style>
    section[data-testid="stSidebar"] {{
        background-color: {COR_PRIMARIA};
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    .main-header {{
        background: linear-gradient(90deg, {COR_PRIMARIA} 0%, #145488 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 6px solid {COR_SECUNDARIA};
    }}
    .main-header h1 {{
        color: black;
        margin: 0;
        font-size: 1.5rem;
    }}
    .main-header p {{
        color: #050203;
        margin: 0.3rem 0 0 0;
    }}
    .status-ok {{
        background-color: rgba(46, 204, 113, 0.15);
        color: #1B8A4B;
        border: 1px solid rgba(46, 204, 113, 0.4);
        padding: 0.4rem 0.7rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}
    .status-error {{
        background-color: rgba(176, 0, 32, 0.12);
        color: #B00020;
        border: 1px solid rgba(176, 0, 32, 0.4);
        padding: 0.4rem 0.7rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>⚙️ Assistente de Manual em PDF (NVIDIA RAG)</h1>
    <p>🏭 PowerTech Solutions &nbsp;•&nbsp; 🔌 CLP PowerLogic PT-3000</p>
</div>
""", unsafe_allow_html=True)

st.write("Faça perguntas em português sobre o produto com base no manual fornecido.")

with st.sidebar:
    st.markdown("## ⚙️ PowerTech Solutions")
    st.markdown("### 🤖 Assistente Técnico Inteligente")
    st.markdown("---")

    st.markdown("#### 📋 Informações do Projeto")
    st.markdown("**🏷️ Produto:** CLP PowerLogic PT-3000")
    st.markdown("**🏢 Empresa fictícia:** PowerTech Solutions")
    st.markdown("**🧠 Modelo utilizado:** `meta/llama-3.1-8b-instruct`")
    st.markdown("**📚 Técnica:** RAG (Retrieval Augmented Generation)")

    st.markdown("---")
    st.markdown("#### 👥 Equipe Responsável")
    st.markdown("- Integrante 1\n- Integrante 2\n- Integrante 3\n- Integrante 4")

    st.markdown("---")
    st.markdown("#### 📊 Status do Sistema")
    status_placeholder = st.empty()

    st.markdown("---")
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.rerun()

nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

if not nvidia_api_key:
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
    except Exception:

        pass

if not nvidia_api_key:
    with status_placeholder.container():
        st.markdown('<div class="status-error">🔴 IA não conectada</div>', unsafe_allow_html=True)
        st.markdown('<div class="status-error">🔴 Manual não vetorizado</div>', unsafe_allow_html=True)
    st.info("Por favor, adicione sua NVIDIA_API_KEY no arquivo .env (local) ou nos Secrets do Streamlit (deploy).", icon="🔑")
    st.stop()

@st.cache_resource(show_spinner="Processando o manual em PDF...")
def inicializar_rag():
    nome_arquivo_pdf = "manual.pdf"

    if not os.path.exists(nome_arquivo_pdf):
        st.error(f"Arquivo '{nome_arquivo_pdf}' não foi encontrado!")
        st.stop()

    loader = PyPDFLoader(nome_arquivo_pdf)
    paginas = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(paginas)

    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        nvidia_api_key=nvidia_api_key,
        model_type="passage"
    )

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})

try:
    retriever = inicializar_rag()
except Exception as e:
    with status_placeholder.container():
        st.markdown('<div class="status-error">🔴 Erro ao processar manual</div>', unsafe_allow_html=True)
    st.error(f"Erro ao processar o manual em PDF: {e}")
    st.stop()

try:
    llm = ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        nvidia_api_key=nvidia_api_key,
        temperature=0.2,
        max_tokens=700
    )
except Exception as e:
    with status_placeholder.container():
        st.markdown('<div class="status-error">🔴 IA não conectada</div>', unsafe_allow_html=True)
    st.error(f"Erro ao conectar com a API da NVIDIA: {e}")
    st.stop()

with status_placeholder.container():
    st.markdown('<div class="status-ok">🟢 Manual carregado</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-ok">🟢 Vetorização concluída</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-ok">🟢 IA conectada</div>', unsafe_allow_html=True)

template_prompt = """
Você é um engenheiro especialista em manutenção industrial, atuando como assistente
técnico da PowerTech Solutions.

Sua tarefa é auxiliar técnicos a diagnosticar problemas, interpretar alarmes e
executar procedimentos de manutenção do equipamento, com base no manual técnico.

As informações de contexto abaixo foram extraídas do manual do produto e ESTÃO EM INGLÊS.
Sua tarefa é analisar o contexto em inglês, mas responder à pergunta do usuário OBRIGATORIAMENTE EM PORTUGUÊS.

Use estritamente as informações fornecidas para responder. As informações devem ser
obtidas exclusivamente do manual técnico. Se a resposta não puder ser encontrada no
texto, diga explicitamente: "Desculpe, mas essa informação não consta no manual do produto."

Contexto (em inglês):
{context}

Pergunta (em português): {question}
Resposta em português:
"""
prompt = ChatPromptTemplate.from_template(template_prompt)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Processei o manual em inglês com sucesso. O que você deseja saber sobre o produto?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt_usuario := st.chat_input("Ex: Qual é o significado do erro 4?"):
    st.session_state.messages.append({"role": "user", "content": prompt_usuario})
    with st.chat_message("user"):
        st.write(prompt_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando manual técnico..."):
            try:
                resposta = rag_chain.invoke(prompt_usuario)
                st.write(resposta)
                st.session_state.messages.append({"role": "assistant", "content": resposta})
            except Exception as e:
                st.error(f"Erro ao processar a requisição na API da NVIDIA: {e}")
