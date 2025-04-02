# --- Importaciones actualizadas y compatibles ---
import pandas as pd
import numpy as np
import asyncio
import sys
from datetime import datetime

# Configurar el event loop para Windows (evita errores en Streamlit)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Preprocesamiento y modelos ---
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from pydantic import BaseModel  # Usar Pydantic V2 directamente
import streamlit as st

# --- Configuración de Streamlit ---
st.set_page_config(page_title="Asistente de Moda IA", layout="wide")
st.title("🎨 Recomendador Personalizado de Moda")

# --- Carga y preparación de datos ---
@st.cache_data
def load_data():
    # Cargar datasets
    customers = pd.read_csv(r"D:\GAMIC\PORTFOLIO\CLOTHING RETAIL - PRS POWERED BY LLM\01_Clothing_Retail_Synthetic_Data_Creation\customers.csv")
    transactions = pd.read_csv(r"D:\GAMIC\PORTFOLIO\CLOTHING RETAIL - PRS POWERED BY LLM\01_Clothing_Retail_Synthetic_Data_Creation\transactions.csv")
    products = pd.read_csv(r"D:\GAMIC\PORTFOLIO\CLOTHING RETAIL - PRS POWERED BY LLM\01_Clothing_Retail_Synthetic_Data_Creation\products.csv")
    interactions = pd.read_csv(r"D:\GAMIC\PORTFOLIO\CLOTHING RETAIL - PRS POWERED BY LLM\01_Clothing_Retail_Synthetic_Data_Creation\interactions.csv")

    # Unir datos
    merged_data = pd.merge(transactions, products, on="product_id")
    merged_data = pd.merge(merged_data, customers, on="customer_id")
    merged_data = pd.merge(
        merged_data, 
        interactions, 
        on=["customer_id", "product_id"], 
        how="left",
        suffixes=("_txn", "_int")
    )

    # Crear descripción semántica de productos
    products["text_description"] = (
        "Categoría: " + products["category"] + ". " +
        "Estilo: " + products["formality"] + ". " +
        "Color: " + products["color"] + ". " +
        "Materiales: " + products["materials"] + ". " +
        "Temporada: " + products["season"]
    )

    return merged_data, products

merged_data, products = load_data()

# --- Generación de embeddings y FAISS ---
@st.cache_resource
def setup_embeddings():
    # Modelo de embeddings en español
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    product_embeddings = model.encode(products["text_description"].tolist())
    products["embedding"] = [emb.tolist() for emb in product_embeddings]
    
    # Crear base vectorial
    embeddings_model = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    vector_db = FAISS.from_texts(
        texts=products["text_description"].tolist(),
        embedding=embeddings_model
    )
    
    return vector_db

vector_db = setup_embeddings()

# --- Configuración de Gemini ---
@st.cache_resource
def setup_gemini():
    return ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key="AIzaSyBO0JURYEXi-up4dvUbjwjonSRpZcB92TU",  # Reemplazar con tu API key
        temperature=0.3,
        convert_system_message_to_human=True  # Corrección 1
    )

llm = setup_gemini()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    return_source_documents=True
)

# --- Lógica de recomendaciones ---
def get_recommendations(customer_id: int):
    try:
        # Validación 1: Cliente existe
        if customer_id not in merged_data["customer_id"].values:
            return "❌ Error: ID de cliente no válido"

        customer_data = merged_data[merged_data["customer_id"] == customer_id]

        # Validación 2: Datos no vacíos
        if customer_data.empty:
            return "⚠️ Cliente sin historial registrado"

        # Validación 3: Última interacción existe
        customer_data_sorted = customer_data.sort_values("event_timestamp", ascending=False)
        if customer_data_sorted.empty:
            return "⚠️ Sin interacciones recientes"
        
        last_interaction = customer_data_sorted.iloc[0]  # Mejor práctica que iloc[-1]

        # Construir contexto
        context = f"""
        Perfil del cliente:
        - Edad: {last_interaction['age']}
        - Género: {last_interaction['gender']}
        - Última interacción: {last_interaction['event_type']}
        - Producto visto: {last_interaction['product_name']}
        """

        # Generar recomendación
        response = qa_chain.invoke({
            "query": f"Recomienda 3 productos relevantes basados en este contexto:{context}. Formato: Lista numerada con justificación técnica en español"
        })["result"]
        
        return response

    except Exception as e:
        return f"🚨 Error interno: {str(e)}"

# --- Interfaz de usuario ---
customer_id = st.number_input("Ingrese su ID de cliente:", min_value=1, step=1)

if st.button("Generar recomendaciones"):
    if customer_id:
        with st.spinner("Analizando tu estilo..."):
            recommendations = get_recommendations(customer_id)
            
            # Mostrar resultados o errores
            if "Error" in recommendations or "⚠️" in recommendations:
                st.error(recommendations)
            else:
                st.success("### Recomendaciones personalizadas")
                st.markdown(recommendations)
                
                # Mostrar historial reciente
                st.divider()
                st.subheader("📚 Tu historial reciente")
                hist_data = merged_data[merged_data["customer_id"] == customer_id][
                    ["product_name", "category", "purchase_date"]
                ].head(3)
                st.dataframe(hist_data, hide_index=True)
    else:
        st.warning("Por favor ingrese un ID válido")

# --- Instrucciones para ejecutar ---
"""
**Para ejecutar:**
1. Instalar dependencias: `pip install -r requirements.txt`
2. Obtener API Key de Google AI Studio
3. Ejecutar: `streamlit run app.py`
"""