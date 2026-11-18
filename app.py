import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Classificador SSP-SP | MLP", page_icon="📱", layout="centered")

@st.cache_resource
def carregar_modelo():
    modelo = tf.keras.models.load_model("modelo_mlp.keras")
    
    # Carrega apenas as listas de categorias (sem depender da versão do sklearn do Colab)
    try:
        prep = joblib.load("preprocessador.joblib")
        bairros = prep['bairros']
        periodos = prep['periodos']
        dias = prep['dias']
        locais = prep['locais']
        marcas = prep['marcas']
    except Exception:
        # Fallback caso o joblib falhe por versão do Python
        import pickle
        with open("preprocessador.joblib", "rb") as f:
            prep = pickle.load(f)
            bairros = prep['bairros']
            periodos = prep['periodos']
            dias = prep['dias']
            locais = prep['locais']
            marcas = prep['marcas']

    # Reconstrói o OneHotEncoder nativamente na versão do Streamlit
    features = ['BAIRRO_TRATADO', 'PERIODO', 'DIA_SEMANA', 'LOCAL_TRATADO', 'MARCA_TRATADA']
    categorias = [bairros, periodos, dias, locais, marcas]
    
    encoder = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(categories=categorias, handle_unknown='ignore', sparse_output=False), features)]
    )
    # Ajusta com uma linha dummy apenas para inicializar o transformador
    df_dummy = pd.DataFrame([[bairros[0], periodos[0], dias[0], locais[0], marcas[0]]], columns=features)
    encoder.fit(df_dummy)

    dados = {
        'encoder': encoder,
        'bairros': bairros,
        'periodos': periodos,
        'dias': dias,
        'locais': locais,
        'marcas': marcas
    }
    return modelo, dados

modelo, prep = carregar_modelo()

st.title("📱 Classificador de Ocorrências de Celulares (SP)")
st.markdown(
    "Informe as características de uma ocorrência registrada na capital paulista para que a rede neural **MLP** "
    "estime a compatibilidade entre **Roubo (art. 157)** e **Furto (art. 155)**."
)

st.divider()

with st.form("form_classificacao"):
    col1, col2 = st.columns(2)
    with col1:
        bairro = st.selectbox("Bairro da Ocorrência", prep['bairros'])
        periodo = st.selectbox("Período do Dia", prep['periodos'])
        marca = st.selectbox("Marca do Celular", prep['marcas'])
    with col2:
        dia = st.selectbox("Dia da Semana", prep['dias'])
        local = st.selectbox("Tipo de Local", prep['locais'])

    btn_analisar = st.form_submit_button("Classificar Ocorrência", use_container_width=True)

if btn_analisar:
    df_entrada = pd.DataFrame([{
        'BAIRRO_TRATADO': bairro,
        'PERIODO': periodo,
        'DIA_SEMANA': dia,
        'LOCAL_TRATADO': local,
        'MARCA_TRATADA': marca
    }])

    X_proc = prep['encoder'].transform(df_entrada)
    prob_roubo = float(modelo.predict(X_proc, verbose=0)[0][0])
    prob_furto = 1.0 - prob_roubo

    st.subheader("Resultado da Análise")

    if prob_roubo >= 0.5:
        st.error(f"**Perfil predominante: ROUBO** ({prob_roubo*100:.1f}%)")
    else:
        st.warning(f"**Perfil predominante: FURTO** ({prob_furto*100:.1f}%)")

    c1, c2 = st.columns(2)
    c1.metric("Compatibilidade com Roubo", f"{prob_roubo*100:.1f}%")
    c2.metric("Compatibilidade com Furto", f"{prob_furto*100:.1f}%")

    st.caption(
        "⚠️ *Nota metodológica:* Estes percentuais refletem apenas a probabilidade condicional estimada pela MLP "
        "com base no histórico da SSP-SP 2026 para uma ocorrência com essas características — não representam risco de ocorrência futura."
    )
