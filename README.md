# Classificação de Ocorrências Criminais em São Paulo (Roubo vs. Furto) com MLP

Projeto desenvolvido para a Global Solution da disciplina de **Aprendizado Profundo com Redes Neurais Convolucionais**. O objetivo é utilizar uma rede neural **Multilayer Perceptron (MLP)** para classificar registros de segurança pública da capital paulista entre **Furto (art. 155)** e **Roubo (art. 157)** com base no contexto espaço-temporal da ocorrência e na marca do aparelho subtraído.

🔗 **Acesse a aplicação web funcionando:** [COLE AQUI O LINK DO SEU APP NO STREAMLIT]

---

## 📊 Base de Dados e Recorte

Os dados foram obtidos no Portal de Estatísticas da **Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP)**:
* **Fonte:** https://www.ssp.sp.gov.br/estatistica/consultas
* **Base utilizada:** Celulares Subtraídos (`CELULAR_2026` — consolidado de janeiro a setembro de 2026, totalizando 256.077 registros).
* **Recorte e Limpeza:**
  * Filtro geográfico pelo local do fato (`CIDADE == 'S.PAULO'`), garantindo apenas ocorrências da capital paulista.
  * Remoção de duplicidades por boletim de ocorrência (`ID_DELEGACIA`, `ANO_BO`, `NUM_BO`).
  * Definição do alvo binário a partir da coluna `RUBRICA` (`0 = Furto`, `1 = Roubo`).
  * Amostragem estratificada de **20.000 ocorrências** únicas (divididas em 70% treino, 15% validação e 15% teste).
* **Atributos de Entrada (Features):** `BAIRRO_TRATADO`, `PERIODO`, `DIA_SEMANA`, `LOCAL_TRATADO` e `MARCA_TRATADA` (codificados via `OneHotEncoder`). Colunas que revelam o enquadramento penal ou dados identificadores foram descartadas para evitar *data leakage*.

---

## 🧠 Arquitetura Escolhida e Resultados

Foram testadas três configurações de MLP (Baseline, Profunda e Regularizada). A arquitetura vencedora (**MLP 3 — Regularizada**) possui:
* **Camada Oculta 1:** 64 neurônios (ativação `ReLU`) + `Dropout(0.3)`
* **Camada Oculta 2:** 32 neurônios (ativação `ReLU`) + `Dropout(0.2)`
* **Camada de Saída:** 1 neurônio (ativação `Sigmoid`)
* **Treinamento:** Otimizador `Adam` (`lr = 0.0005`), perda `Binary Crossentropy`, `batch_size = 64`, `30 épocas`.

### Desempenho no Conjunto de Teste (Dados Inéditos)
* **Acurácia:** 70,30%
* **Precision:** 0,5769
* **Recall:** 0,5465
* **F1-Score:** 0,5613

*(Nota: O modelo estima a compatibilidade condicional de uma ocorrência já registrada com os padrões históricos de roubo ou furto, não calculando risco futuro de crime).*

---

## 📁 Estrutura do Repositório

```text
GS2.2026-AprendizadoProfundo/
├── app.py                     # Aplicação web interativa desenvolvida em Streamlit
├── modelo_mlp.keras           # Modelo MLP vencedor treinado e exportado
├── categorias.json            # Categorias mapeadas para o One-Hot Encoding da interface
├── requirements.txt           # Dependências do projeto
├── GS2.2026-AprendizadoProfundo-Celulares.ipynb # Jupyter Notebook documentado com todo o pipeline e experimentos
└── README.md                  # Documentação do projeto
