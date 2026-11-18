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
import streamlit as st
import pandas as pd
import json
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Classificador SSP-SP | MLP", page_icon="📱", layout="centered")

@st.cache_resource
def carregar_modelo():
    modelo = tf.keras.models.load_model("modelo_mlp.keras")
    
    with open("categorias.json", "r", encoding="utf-8") as f:
        cats = json.load(f)

    features = ['BAIRRO_TRATADO', 'PERIODO', 'DIA_SEMANA', 'LOCAL_TRATADO', 'MARCA_TRATADA']
    categorias_ordenadas = [
        cats['bairros'],
        cats['periodos'],
        cats['dias'],
        cats['locais'],
        cats['marcas']
    ]

    encoder = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(categories=categorias_ordenadas, handle_unknown='ignore', sparse_output=False), features)]
    )
    
    df_dummy = pd.DataFrame([[
        cats['bairros'][0],
        cats['periodos'][0],
        cats['dias'][0],
        cats['locais'][0],
        cats['marcas'][0]
    ]], columns=features)
    encoder.fit(df_dummy)

    return modelo, encoder, cats

modelo, encoder, cats = carregar_modelo()

st.title("📱 Classificador de Ocorrências de Celulares (SP)")
st.markdown(
    "Informe as características de uma ocorrência registrada na capital paulista para que a rede neural **MLP** "
    "estime a compatibilidade entre **Roubo (art. 157)** e **Furto (art. 155)**."
)

st.divider()

with st.form("form_classificacao"):
    col1, col2 = st.columns(2)
    with col1:
        bairro = st.selectbox("Bairro da Ocorrência", cats['bairros'])
        periodo = st.selectbox("Período do Dia", cats['periodos'])
        marca = st.selectbox("Marca do Celular", cats['marcas'])
    with col2:
        dia = st.selectbox("Dia da Semana", cats['dias'])
        local = st.selectbox("Tipo de Local", cats['locais'])

    btn_analisar = st.form_submit_button("Classificar Ocorrência", use_container_width=True)

if btn_analisar:
    df_entrada = pd.DataFrame([{
        'BAIRRO_TRATADO': bairro,
        'PERIODO': periodo,
        'DIA_SEMANA': dia,
        'LOCAL_TRATADO': local,
        'MARCA_TRATADA': marca
    }])

    X_proc = encoder.transform(df_entrada)
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

    st.caption(
        "⚠️ *Nota metodológica:* Estes percentuais refletem apenas a probabilidade condicional estimada pela MLP "
        "com base no histórico da SSP-SP 2026 para uma ocorrência com essas características — não representam risco de ocorrência futura."
    )
