import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag.vectorstore import get_vectorstore 

def get_qa_chain(url: str):
    # 1. Inisialisasi LLM
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 2. Ambil Retriever
    vectorstore = get_vectorstore(url)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Definisikan Prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Format dokumen hasil retrieve agar menjadi teks utuh
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Buat Chain dengan LCEL murni (Bypass modul chains)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
