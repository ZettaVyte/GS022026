# 🐢 Plataforma Acredite — CitizenTech de Resgate Cognitivo & Ponte Comunitária

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Protótipo-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Global%20Solution-FIAP-ed145b?style=flat)

Protótipo desenvolvido para a **Global Solution FIAP (CitizenTech)**. O **Acredite** propõe uma jornada de baixa fricção para combater o "sequestro biológico" do córtex pré-frontal causado por escassez crônica, insegurança alimentar e exaustão laboral, unindo triagem supervisionada por Machine Learning, direcionamento territorial e acompanhamento de micro-hábitos.

> **Nota de Transparência:** Este repositório contém o **MVP local e o pipeline de Machine Learning** desenvolvidos para fins acadêmicos e validação conceitual. Os dados utilizados são **simulados** (parametrizados em escalas científicas reais) e o plano de ação exibido no protótipo utiliza um catálogo local pré-cadastrado. Recursos de nuvem, IA Generativa e integração com dados reais fazem parte da arquitetura conceitual prevista para etapas futuras (ver seção *Roadmap*).

---

## 🎯 O Problema e a Proposta de Solução

Jovens desconectados de estudo e trabalho (*Geração Nem-Nem*) e trabalhadores informais em áreas urbanas, rurais e ribeirinhas frequentemente vivem sob estresse contínuo e privação básica. Nesse estado de sobrevivência, o cérebro sofre de **fadiga de decisão**, dificultando o acesso a serviços públicos e o planejamento de médio/longo prazo.

A proposta do **Acredite** é reduzir drasticamente o atrito inicial:
1. **Triagem Rápida (5 perguntas binárias):** Coleta sinais essenciais baseados em instrumentos validados — **EBIA** (insegurança alimentar), **SRQ-20** (fadiga e sono) e **WHO-5** (energia e bem-estar) — combinados a idade e zona de moradia.
2. **Classificação via Machine Learning:** Um modelo supervisionado classifica o cidadão em três níveis de prioridade:
   * `0 — Urgência Fisiológica`: Prioridade máxima para resgate básico (alimentação, saúde, CRAS/UBS).
   * `1 — Alerta Social`: Suporte comunitário e início da jornada de micro-hábitos.
   * `2 — Pronto para Alocação Produtiva`: Encaminhamento direto para vagas locais, capacitação e cooperativas.
3. **Plano de Ação Territorial (Simulado no MVP):** Com base na classe prevista e no território selecionado, o protótipo exibe uma rota imediata em 3 frentes (homeostase/comunidade, cuidado/capacitação e órgãos públicos de apoio).
4. **Monitor de Micro-Hábitos (8 Semanas):** Interface de check-in diário (*"Sim/Não"*) com foco em pequenas vitórias (sono, redução de hiperestímulo, atividade física e conexão presencial) para apoiar a retomada gradual de autonomia.

---

## 🛠️ Tecnologias Efetivamente Utilizadas neste Repositório

Todo o código presente neste projeto roda localmente em Python, sem dependências de serviços em nuvem ou APIs pagas:

| Categoria | Ferramenta | Uso Real no Projeto |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.11 | Desenvolvimento do pipeline de dados, modelagem e interface. |
| **Manipulação e Análise** | Pandas, NumPy | Geração do dataset simulado, manipulação tabular e estatísticas. |
| **Machine Learning** | Scikit-Learn | Pré-processamento (`ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`), validação cruzada (`StratifiedKFold`), otimização (`GridSearchCV`) e modelos supervisionados. |
| **Visualização (Notebook)** | Matplotlib, Seaborn | Geração dos gráficos de comparação entre modelos, matriz de confusão e importância das variáveis. |
| **Protótipo Interativo** | Streamlit | Construção do aplicativo web local (`app_acredite.py`) com as visões do Cidadão, Monitor de Hábitos e Painel do Gestor. |
| **Ambiente e Gestão** | Jupyter Notebook / VS Code / Trello | Experimentação documentada em células, versionamento e organização de tarefas em Kanban. |

---

## 📊 Transparência sobre os Dados e Modelagem

### Base de Dados (`dataset_acredite.csv`)
Por limitações de acesso a microdados individuais anonimizados que cruzassem simultaneamente fome, saúde mental e território, foi construída uma **base simulada plausível** ($n = 3.000$ registros, `random_state=23`), contendo:
* **Variáveis Preditoras:** `Idade`, `Zona_Moradia` (`Urbana`, `Rural`, `Ribeirinha`), `Trabalho_Precario`, `Fome_EBIA`, `Fadiga_SRQ20`, `Apatia_WHO5` e `Isolamento_Social`.
* **Realismo Estatístico:** Regras de interação não-lineares (ex.: agravamento da privação alimentar em extremos de idade e combinação de isolamento geográfico com trabalho precário), ruído gaussiano em casos limítrofes e **3% de valores ausentes (`NaN`)** simulando falhas de preenchimento ou conectividade.
* **Variável-Alvo:** `Classe_Intervencao` (`0`, `1` ou `2`).

### Metodologia e Prevenção de *Data Leakage*
* Divisão estratificada (**75% treino / 25% teste**) executada **antes** de qualquer imputação, escala ou codificação.
* Todo o pré-processamento foi encapsulado em `ColumnTransformer` + `Pipeline` e ajustado exclusivamente dentro dos *folds* de treino durante a validação cruzada (`StratifiedKFold` com 5 divisões).
* Foram treinados e comparados três modelos supervisionados (**Regressão Logística**, **Decision Tree** e **Random Forest**), com ajuste de hiperparâmetros via `GridSearchCV` no Random Forest (`n_estimators=150`, `max_depth=None`, `min_samples_leaf=3`, `class_weight='balanced'`).

### Resultados no Conjunto de Teste (750 amostras)

| Modelo Supervisionado | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Recall Classe 0 (Urgência) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Decision Tree | 74% | 0.79 | 0.73 | 0.75 | 65% |
| Regressão Logística | 81% | 0.81 | 0.82 | 0.81 | 86% |
| **Random Forest (Tuned)** 🏆 | **85%** | **0.85** | **0.86** | **0.86** | **91%** |

> **Critério de Escolha:** Na triagem social, o erro mais crítico é o **Falso Negativo na Classe 0 (Urgência Fisiológica)** — deixar de identificar alguém em privação alimentar e exaustão severa. O **Random Forest otimizado** entregou o melhor equilíbrio geral (`F1-Macro = 0.86`) e o maior **Recall na Classe 0 (91%)**, minimizando a invisibilidade de casos extremos.

---

## 📂 Estrutura do Repositório

```text
├── GS_2TSCOA_ML_Acredite.ipynb   # Notebook completo com os 9 itens do pipeline de ML
├── dataset_acredite.csv          # Base simulada plausível gerada pelo notebook (n = 3.000)
├── app_acredite.py               # Protótipo interativo em Streamlit (3 abas)
├── comparacao_modelos.png        # Gráfico comparativo entre os modelos testados
├── matriz_confusao_rf.png        # Matriz de confusão do Random Forest no conjunto de teste
├── feature_importance_rf.png     # Importância relativa das variáveis (Gini Importance)
└── README.md                     # Documentação do projeto
```

<<<<<<< HEAD
# 🚀 Como Executar Localmente
1. Pré-requisitos
Recomenda-se **Python 3.11.x** em um ambiente virtual isolado (*venv* ou *pyenv*).

=======
🚀 Como Executar Localmente
1. Pré-requisitos
Recomenda-se **Python 3.11.x** em um ambiente virtual isolado (*venv* ou *pyenv*).

Bash
>>>>>>> 8076a04960f223c091837804454ad29b8642f71c
# Clone o repositório
`git clone [https://github.com/](https://github.com/)ZettaVyte/GS022026.git`
`cd GS022026`

# Crie e ative o ambiente virtual
`python -m venv .venv`
`source .venv/bin/activate  # Linux / WSL / Mac`
# .venv\Scripts\activate   # Windows

# Instale as dependências
`pip install pandas numpy scikit-learn matplotlib seaborn streamlit jupyter`

2. Executar o Notebook de Machine Learning
Bash
`jupyter notebook GS_2TSCOA_ML_Acredite.ipynb`

3. Executar o Protótipo em Streamlit
`streamlit run app_acredite.py`

O protótipo abre no navegador com três abas funcionais:

1. **Triagem & Plano de Ação (Cidadão):** Classificação em tempo real pelo Random Forest + exibição de plano de ação simulado para dois cenários territoriais (Maués - AM e Manaus - AM).

2. **Monitor de Micro-Hábitos (8 Semanas):** Check-in diário interativo e curva ilustrativa de consolidação de hábitos.

3. **Painel Analítico (Gestor & ML):** KPIs do dataset, importância das variáveis, matriz de confusão e distribuição por zona de moradia.

# 🗺️ Limitações Atuais e Roadmap (Próximos Passos)
Como esta entrega representa uma prova de conceito acadêmica executada localmente, os seguintes itens ainda **não foram implementados em código**, mas compõem a evolução planejada para levar o **Acredite** a produção:

* **Substituição dos Dados Simulados por Dados Reais:** Integração com bases públicas oficiais (IBGE, DATASUS, CadÚnico e OpenStreetMap) e coleta validada em campo em parceria com agentes comunitários/CRAS para retreinar o modelo com dados reais.

* **Infraestrutura em Nuvem (Oracle Cloud - OCI):** Migração do armazenamento e processamento para OCI Object Storage, Oracle Autonomous Database, OCI Data Integration e orquestração de retreinamento com Apache Airflow, expondo o modelo via FastAPI.

* **Integração de IA Generativa (LLM + RAG):** Substituição do catálogo estático do protótipo por um agente de IA (OCI Generative AI + AI Vector Search) capaz de consultar equipamentos públicos atualizados e gerar orientações personalizadas em linguagem acessível.

**Canais de Baixa Conectividade e BI em Escala:** Disponibilização da triagem via WhatsApp/SMS leve (além do Modo Agente presencial) e dashboards territoriais em produção via Tableau.

# 🔒 Ética, Vieses e Uso Responsável
* **Viés de Conectividade:** Coletas puramente digitais tendem a sub-representar populações rurais, ribeirinhas ou sem smartphone. O protótipo já prevê conceitualmente o Modo Agente Comunitário (busca ativa presencial) e canais leves.

* **Viés de Autodeclaração:** Pessoas em vulnerabilidade extrema podem omitir a fome por constrangimento; por isso o modelo combina indicadores indiretos (Fadiga_SRQ20, Apatia_WHO5, Isolamento_Social).

* **Privacidade (LGPD):** Em uma implementação real, todos os registros analíticos devem ser anonimizados e restritos à finalidade de acolhimento e planejamento territorial.

# 👥 Autoria
Projeto desenvolvido para a **Global Solution 022026 FIAP — Turma 2TSCOA**.

<<<<<<< HEAD
* Nome: **VITOR "ZETTAVYTE" ANTUNES**
=======
* Nome: VITOR "ZETTAVYTE" ANTUNES
>>>>>>> 8076a04960f223c091837804454ad29b8642f71c
* Vídeo Pitch: <Inserir link do vídeo>