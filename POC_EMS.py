import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

# 1. Carrega variáveis de ambiente
load_dotenv()

# 2. Diretórios e arquivos
# pdf_dir = "C:\\Users\\loren\\Documents\\PSW_local\\Clientes\\EMS\\Bulas"
# csv_path = "C:\\Users\\loren\\Documents\\PSW_local\\Clientes\\EMS\\Venda\\Venda Medicamentos Novo.csv"

pdf_dir = "data/pdfs"
csv_path = "data/Venda Medicamentos Novo.csv"

# 3. Carrega dados de vendas
df_venda = pd.read_csv(csv_path, dtype={"Medicamento": str}, decimal=',')
df_venda["Preço de Venda"] = df_venda["Preço de Venda"].astype(float)
df_venda["Quantidade Estoque"] = df_venda["Quantidade Estoque"].astype(int)

# 4. Inicializa embeddings, LLM e vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
llm = ChatOpenAI(model="gpt-4", temperature=0.4)

# 5. Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)

# 6. Carregamento dos PDFs
medicamentos_disponiveis = set()

for file_name in os.listdir(pdf_dir):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(pdf_dir, file_name)
        medicamento = os.path.splitext(file_name)[0].strip()
        medicamentos_disponiveis.add(medicamento)

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["Medicamento"] = medicamento

        splits = text_splitter.split_documents(docs)
        for split in splits:
            split.metadata["Medicamento"] = medicamento

        vector_store.add_documents(splits)

# 7. Cria prompt template
prompt = ChatPromptTemplate.from_template(
    "Você é um assistente especializado em bulas de remédios e informações de venda de medicamentos.\n"
    "Use o contexto abaixo para responder à pergunta do usuário de forma clara e precisa.\n\n"
    "Você possui duas fontes de dados:\n"
    "- Bulas dos medicamentos disponíveis.\n"
    "- Tabela comercial contendo preço de venda, margem de lucro e estoque dos medicamentos.\n\n"
    "Regras:\n"
    "1. Sempre recomende um medicamento presente na base de conhecimento (bulas).\n"
    "2. O medicamento recomendado deve ser diferente do que o paciente já está tomando.\n"
    "3. O medicamento recomendado deve tratar a enfermidade informada.\n"
    "4. Sempre recomende todos os medicamentos existentes na base de conhecimento que podem tratar a enfermidade.\n"
    "5. Se o paciente já estiver tomando o medicamento ideal, não o recomende novamente.\n"
    "6. Ao recomendar um medicamento, verifique se há contraindicações com outros medicamentos que o paciente já está tomando.\n"
    "7. Use apenas o conteúdo das bulas como fonte para justificar recomendações clínicas.\n"
    "8. Você poderá usar a tabela de vendas se precisar explicar sobre aspectos como preço, margem, estoque, ou questões comerciais.\n"
    "9. Caso falte alguma informação, peça ao usuário para completá-la.\n"
    "10. Considere que você esteja se referindo a um médico, portanto, sob nenhum pretexto será necessário adverter sobre a necessidade de aconselhamento de um médico.\n"
    "11. Considere o contexto clínico do paciente (ex: gravidez, idade, comorbidades) para fornecer informações detalhadas sobre riscos potenciais de uso do medicamento recomendado.\n"
    "12. Sempre apresentar informações comerciais ao sugerir o medicamento recomendado, seguindo o seguinte formato padronizado:\n"
    "- Preço: sempre formatar como R$10.00 (duas casas decimais e símbolo R$).\n"
    "- Estoque: sempre exibir apenas o número de unidades, como 25 unidades.\n"
    "13. Considere que o ideal é recomendar um medicamento de maior valor de venda.\n"
    "14. Sempre explique as informações da forma mais completa possível a partir de aspectos teóricos existentes nas bulas, evitando respostas vagas ou incompletas.\n\n"

    "Contexto do paciente:\n"
    "- Enfermidade: {enfermidade}\n"
    "- Medicamento atual: {medicamento_atual}\n"
    "- Contexto clínico adicional: {contexto_paciente}\n\n"
    "Lista de medicamentos disponíveis: {lista_medicamentos}\n\n"
    "Informações das bulas:\n{context}\n\n"
    "Tabela de vendas (se necessário):\n{context_vendas}\n\n"
    "Histórico da conversa:\n{history}\n\n"
    "Pergunta: {question}"
)

# 8. Interface Streamlit com histórico estilo chat
st.set_page_config(page_title="Agente de Recomendação de Medicamentos", layout="centered")
st.title("💊 Agente de Recomendação de Medicamentos")

# Inicializa o histórico se ainda não existir
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Contexto do paciente")
    enfermidade = st.text_input("Enfermidade a tratar", st.session_state.get("enfermidade", ""))
    medicamento_atual = st.text_input("Medicamento atual", st.session_state.get("medicamento_atual", ""))
    contexto_paciente = st.text_area("Contexto clínico adicional", st.session_state.get("contexto_paciente", ""))

    # Atualiza sessão com contexto persistente
    st.session_state["enfermidade"] = enfermidade
    st.session_state["medicamento_atual"] = medicamento_atual
    st.session_state["contexto_paciente"] = contexto_paciente


# Renderiza mensagens anteriores
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

# Campo de input de nova pergunta
pergunta = st.chat_input("Digite sua pergunta")

if pergunta:
    st.chat_message("user").markdown(pergunta)
    st.session_state.history.append({"role": "user", "content": pergunta})

    # Monta texto da conversa anterior para o prompt
    historico_formatado = ""
    for h in st.session_state.history:
        if h["role"] == "user":
            historico_formatado += f"Usuário: {h['content']}\n"
        else:
            historico_formatado += f"IA: {h['content']}\n"

    # Gera texto da base de vendas
    dados_venda_texto = ""
    for _, row in df_venda.iterrows():
        dados_venda_texto += (
            f"- {row['Medicamento'].capitalize()}: Preço R${row['Preço de Venda']:.2f}, "
            f"Estoque {row['Quantidade Estoque']} unidades.\n"
        )

    # Recuperação de contexto das bulas
    results = vector_store.similarity_search(pergunta, k=5)

    if results:
        context_parts = []
        for doc in results:
            nome_remedio = doc.metadata.get("Medicamento", "Desconhecido")
            context_parts.append(f"[{nome_remedio}] {doc.page_content}")

        context = "\n\n".join(context_parts)
        lista_medicamentos = ", ".join(sorted(medicamentos_disponiveis))

        final_prompt = prompt.format_messages(
            context=context,
            context_vendas=dados_venda_texto,
            history=historico_formatado,
            question=pergunta,
            enfermidade=enfermidade,
            medicamento_atual=medicamento_atual,
            contexto_paciente=contexto_paciente,
            lista_medicamentos=lista_medicamentos
        )

        resposta = llm.invoke(final_prompt)
        st.chat_message("assistant").markdown(resposta.content)
        st.session_state.history.append({"role": "assistant", "content": resposta.content})
    else:
        st.warning("Nenhum contexto encontrado nas bulas.")