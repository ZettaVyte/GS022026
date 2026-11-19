import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st  # type: ignore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA E ESTILO VISUAL (PALETA ACREDITE)
# ==============================================================================
st.set_page_config(
    page_title="Acredite | CitizenTech",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #164e63 0%, #115e59 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .card-urgencia {
        background-color: #fef2f2;
        border-left: 8px solid #dc2626;
        padding: 1.2rem;
        border-radius: 8px;
        color: #7f1d1d;
    }
    .card-alerta {
        background-color: #fffbeb;
        border-left: 8px solid #d97706;
        padding: 1.2rem;
        border-radius: 8px;
        color: #78350f;
    }
    .card-pronto {
        background-color: #ecfdf5;
        border-left: 8px solid #059669;
        padding: 1.2rem;
        border-radius: 8px;
        color: #064e3b;
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CARREGAMENTO DO DATASET E TREINAMENTO EM CACHE DO MODELO CAMPEÃO
# ==============================================================================
@st.cache_resource
def carregar_dados_e_treinar():
    csv_path = "dados_acredite.csv"
    df = pd.read_csv(csv_path)

    X = df.drop('Classe_Intervencao', axis=1)
    y = df['Classe_Intervencao']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    col_num = ['Idade']
    col_bin = ['Trabalho_Precario', 'Fome_EBIA', 'Fadiga_SRQ20', 'Apatia_WHO5', 'Isolamento_Social']
    col_cat = ['Zona_Moradia']

    preprocessador = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), col_num),
        ('bin', Pipeline([('imp', SimpleImputer(strategy='most_frequent'))]), col_bin),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))]), col_cat)
    ])

    pipe = Pipeline([
        ('preprocessamento', preprocessador),
        ('modelo', RandomForestClassifier(
            n_estimators=150, max_depth=None, min_samples_leaf=3,
            class_weight='balanced', random_state=23
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metricas = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'recall_urgencia': recall_score(y_test, y_pred, labels=[0], average=None)[0], # type: ignore
        'cm': confusion_matrix(y_test, y_pred)
    }

    nomes_features = (
        col_num + col_bin +
        list(pipe.named_steps['preprocessamento'].named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(col_cat))
    )
    importancias = pipe.named_steps['modelo'].feature_importances_
    df_imp = pd.DataFrame({'Variável': nomes_features, 'Importância': importancias}).sort_values('Importância', ascending=True)

    return df, pipe, metricas, df_imp

df_base, modelo_rf, metricas_rf, df_importancia = carregar_dados_e_treinar()

# Catálogo simulado de Varredura Web Local por Território
CATALOGO_TERRITORIAL = {
    "Maués - AM (Interior / Ribeirinho)": {
        0: {
            "homeostase": "CRAS Maués (Bairro Ramalho Júnior) & Cozinha Comunitária Municipal — Entrega imediata de refeição e cesta emergencial sem agendamento.",
            "saude": "UBS Fluvial / Básica e CAPS Maués — Acolhimento para exaustão severa e insônia.",
            "cobranca": "Secretaria Municipal de Assistência Social (SEMAS) e Defensoria Pública — Protocolo automático de vulnerabilidade alimentar."
        },
        1: {
            "comunidade": "Roda de Apoio Jovem & Pastoral Comunitária (Bairro Maresia) — Encontros terça e quinta às 18h.",
            "capacitacao": "Oficina Prática de Beneficiamento Agroflorestal e Digitalização (CETEC Maués).",
            "cobranca": "Conselho Municipal da Juventude — Solicitação de bolsa-auxílio permanência."
        },
        2: {
            "trabalho": "Cooperativa de Produtores de Guaraná & Comércio Local — 14 vagas abertas de Jovem Aprendiz e logística leve.",
            "comunidade": "Grupo de Corrida, Calistenia e Mentoria Comunitária na Orla de Maués (Sábados, 7h).",
            "cobranca": "SINE Municipal / Sala do Empreendedor — Microcrédito orientado e emissão de carteira."
        }
    },
    "Manaus - AM (Periferia Urbana)": {
        0: {
            "homeostase": "Prato Cheio (Restaurante Popular - Zona Leste/Norte) & CRAS mais próximo a 800m via GPS.",
            "saude": "UBS da Família & Policlínica — Atendimento prioritário sem burocracia.",
            "cobranca": "SEMMASDH Manaus — Solicitação de Auxílio Estadual e Tarifa Social de Água/Energia."
        },
        1: {
            "comunidade": "Coletivo Jovem Periferia Viva & Centro Cultural Comunitário — Oficinas e escuta ativa.",
            "capacitacao": "CETAM / SENAI Manaus — Cursos rápidos de 4 semanas com auxílio-transporte.",
            "cobranca": "Ouvidoria SUS / Assistência Social — Monitoramento de fila de atendimento."
        },
        2: {
            "trabalho": "Polo Digital de Manaus & Cooperativas de Reciclagem/Logística — Vagas imediatas para 1º emprego.",
            "comunidade": "Rede de Empreendedores de Bairro & Mentoria de Carreira.",
            "cobranca": "SINE Manaus — Encaminhamento direto para entrevista."
        }
    }
}

# ==============================================================================
# BARRA LATERAL (MODO DE ACESSIBILIDADE E TERRITÓRIO)
# ==============================================================================
with st.sidebar:
    st.markdown("## 🐢 Aplicativo **Acredite**\nCriado por Vitor \"ZettaVyte\" Antunes")
    st.divider()
    modo_operacao = st.radio(
        "Modo de Acesso (Inclusão Digital):",
        ["📱 Modo Cidadão (App / WhatsApp Leve)", "🤝 Modo Agente Comunitário (Busca Ativa)"]
    )
    territorio_sel = st.selectbox(
        "Território Mapeado pelo Agente:",
        list(CATALOGO_TERRITORIAL.keys())
    )
    acessibilidade_voz = st.toggle("🔊 Leitura por Voz e Alto Contraste", value=True)
    st.divider()
    st.info(
        "🔒 **Conformidade LGPD:** Dados anonimizados e processados apenas para "
        "triagem de homeostase e conexão comunitária."
    )

# ==============================================================================
# CABEÇALHO PRINCIPAL
# ==============================================================================
st.markdown("""
<div class="main-header">
    <h2 style="margin:0;">🐢 Aplicativo Acredite — CitizenTech de Resgate Cognitivo</h2>
    <p style="margin:4px 0 0 0; opacity:0.9;">
        Triagem Supervisionada (Machine Learning) + Agente de Varredura Territorial + Monitor de Micro-Hábitos (8 Semanas)
    </p>
</div>
""", unsafe_allow_html=True)

aba_cidadao, aba_habitos, aba_gestor = st.tabs([
    "🧭 1. Triagem & Plano de Ação (Cidadão)",
    "📆 2. Monitor de Micro-Hábitos (8 Semanas)",
    "📊 3. Painel Analítico (Gestor & ML)"
])

# ==============================================================================
# ABA 1: VISÃO DO CIDADÃO (TRIAGEM RÁPIDA + PLANO DE AÇÃO DO AGENTE)
# ==============================================================================
with aba_cidadao:
    col_form, col_res = st.columns([1, 1.15], gap="large")

    with col_form:
        st.subheader("Etapa 1 — Triagem Rápida (Sem Burocracia)")
        st.caption("Apenas 5 perguntas binárias baseadas nas escalas oficiais EBIA, SRQ-20 e WHO-5.")

        c1, c2 = st.columns(2)
        with c1:
            idade_input = st.slider("Idade do Cidadão:", 15, 65, 21)
        with c2:
            zona_input = st.selectbox("Zona de Moradia:", ["Urbana", "Rural", "Ribeirinha"])

        p1_trab = st.radio(
            "1. No último mês, esteve sem trabalho fixo ('Nem-Nem') ou viveu apenas de bicos?",
            ["SIM", "NÃO"], horizontal=True, index=0
        )
        p2_fome = st.radio(
            "2. [EBIA] Nos últimos 3 meses, reduziu ou pulou refeições porque o dinheiro acabou?",
            ["SIM", "NÃO"], horizontal=True, index=0
        )
        p3_fadiga = st.radio(
            "3. [SRQ-20] Nos últimos 30 dias, tem dormido mal e se sentido exausto o tempo todo?",
            ["SIM", "NÃO"], horizontal=True, index=0
        )
        p4_apatia = st.radio(
            "4. [WHO-5] Nas últimas 2 semanas, faltou energia ou ânimo para as tarefas do dia?",
            ["SIM", "NÃO"], horizontal=True, index=0
        )
        p5_isol = st.radio(
            "5. [Conexão] Sente que não participa de nenhum grupo, comunidade ou rede de apoio local?",
            ["SIM", "NÃO"], horizontal=True, index=0
        )

    with col_res:
        st.subheader("Etapa 2 — Diagnóstico ML & Rota do Agente IA")

        df_usuario = pd.DataFrame([{
            'Idade': float(idade_input),
            'Zona_Moradia': zona_input,
            'Trabalho_Precario': 1.0 if p1_trab == "SIM" else 0.0,
            'Fome_EBIA': 1.0 if p2_fome == "SIM" else 0.0,
            'Fadiga_SRQ20': 1.0 if p3_fadiga == "SIM" else 0.0,
            'Apatia_WHO5': 1.0 if p4_apatia == "SIM" else 0.0,
            'Isolamento_Social': 1.0 if p5_isol == "SIM" else 0.0
        }])

        classe_pred = int(modelo_rf.predict(df_usuario)[0])
        probs = modelo_rf.predict_proba(df_usuario)[0]
        rotas = CATALOGO_TERRITORIAL[territorio_sel][classe_pred]

        if classe_pred == 0:
            st.markdown(f"""
            <div class="card-urgencia">
                <h3 style="margin:0;">🚨 CLASSIFICAÇÃO: 0 — URGÊNCIA FISIOLÓGICA (Confiança: {probs[0]*100:.1f}%)</h3>
                <p style="margin:6px 0 0 0;">
                    <strong>Diagnóstico Neurobiológico:</strong> Alarme de sobrevivência ativado (sequestro do córtex pré-frontal).
                    Prioridade total é desligar a fome e a exaustão antes de exigir qualquer esforço produtivo.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### 📍 Plano de Ação Imediato (Gerado pelo Agente Local):")
            st.success(f"🍲 **1. Passo Hoje (Homeostase):** {rotas['homeostase']}")
            st.info(f"🏥 **2. Cuidado Físico e Sono:** {rotas['saude']}")
            st.warning(f"⚖️ **3. Quem Cobrar (Direito Cidadão):** {rotas['cobranca']}")

        elif classe_pred == 1:
            st.markdown(f"""
            <div class="card-alerta">
                <h3 style="margin:0;">⚠️ CLASSIFICAÇÃO: 1 — ALERTA SOCIAL / ZONA CINZENTA (Confiança: {probs[1]*100:.1f}%)</h3>
                <p style="margin:6px 0 0 0;">
                    <strong>Diagnóstico Neurobiológico:</strong> Fadiga de decisão e risco de isolamento/hiperestímulo.
                    Momento de inserir em uma comunidade-âncora e iniciar a reprogramação de 8 semanas.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### 📍 Plano de Ação Imediato (Gerado pelo Agente Local):")
            st.success(f"🤝 **1. Ponte Comunitária (A Âncora):** {rotas['comunidade']}")
            st.info(f"🛠️ **2. Micro-Responsabilidade Assistida:** {rotas['capacitacao']}")
            st.warning(f"⚖️ **3. Quem Cobrar (Direito Cidadão):** {rotas['cobranca']}")

        else:
            st.markdown(f"""
            <div class="card-pronto">
                <h3 style="margin:0;">✅ CLASSIFICAÇÃO: 2 — PRONTO PARA ALOCAÇÃO PRODUTIVA (Confiança: {probs[2]*100:.1f}%)</h3>
                <p style="margin:6px 0 0 0;">
                    <strong>Diagnóstico Neurobiológico:</strong> Homeostase preservada e janela de lucidez ativa.
                    Direcionamento direto para trabalho, renda e fortalecimento comunitário.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("#### 📍 Plano de Ação Imediato (Gerado pelo Agente Local):")
            st.success(f"💼 **1. Rito de Passagem (Trabalho Local):** {rotas['trabalho']}")
            st.info(f"🌱 **2. Novo Ambiente de Padrão Elevado:** {rotas['comunidade']}")
            st.warning(f"⚖️ **3. Apoio Institucional:** {rotas['cobranca']}")

        st.caption("📲 *Resumo enviado automaticamente via WhatsApp/SMS leve (consumo zero de franquia).*")

# ==============================================================================
# ABA 2: MONITOR DE MICRO-HÁBITOS (8 SEMANAS DE NEUROPLASTICIDADE)
# ==============================================================================
with aba_habitos:
    st.subheader("📆 Jornada de 8 Semanas — Reprogramação de Hábitos Angulares")
    st.write(
        "Mudanças estruturais no cérebro (córtex cingulado médio anterior e hipocampo) exigem cerca de "
        "**8 semanas de constância**. O Agente realiza check-ins binários diários ('Sim/Não') sem fricção."
    )

    col_h1, col_h2 = st.columns([1.1, 1], gap="large")
    with col_h1:
        st.markdown("#### Check-in Diário do Cidadão (Semana 3 / Dia 4)")
        h1 = st.checkbox("☀️ **Sono e Luz Natural:** Acordei no horário planejado e tomei 10 min de luz solar", value=True)
        h2 = st.checkbox("📵 **Redução de Hiperestímulo:** Fiquei as primeiras 2h do dia longe de vídeos curtos/jogos", value=True)
        h3 = st.checkbox("🏃 **Esforço Físico com Objetivo:** Realizei 20 min de caminhada, calistenia ou trabalho manual", value=True)
        h4 = st.checkbox("🤝 **Conexão Real (Oposto do Vício):** Conversei presencialmente com alguém da comunidade/oficina", value=False)

        concluidos = sum([h1, h2, h3, h4])
        progresso_semana = int((concluidos / 4) * 100)
        st.progress(progresso_semana / 100, text=f"Progresso de Hoje: {concluidos}/4 micro-hábitos cumpridos ({progresso_semana}%)")

        if concluidos >= 3:
            st.success("🐢 **Mensagem do Agente Acredite:** 'Pequenas escolhas desconfortáveis hoje constroem vantagens desproporcionais amanhã. Sua comunidade conta com você!'")
        else:
            st.info("🐢 **Mensagem do Agente Acredite:** 'Sem culpa. Foque apenas em cumprir 1 micro-hábito nas próximas 2 horas.'")

    with col_h2:
        st.markdown("#### Evolução Neuroplástica Acumulada (8 Semanas)")
        semanas = [f"Sem {i}" for i in range(1, 9)]
        adesao = [45, 58, 72, 78, 82, 85, 90, 94]
        fig_h, ax_h = plt.subplots(figsize=(6, 3.2))
        ax_h.plot(semanas, adesao, marker='o', linewidth=2.5, color='#0f766e')
        ax_h.fill_between(semanas, adesao, color='#0f766e', alpha=0.15)
        ax_h.set_ylim(0, 105)
        ax_h.set_ylabel("% Check-ins Positivos")
        ax_h.set_title("Consolidação do Hábito e Queda de Recaída")
        st.pyplot(fig_h)

# ==============================================================================
# ABA 3: PAINEL ANALÍTICO DO GESTOR PÚBLICO E RESULTADOS DO MACHINE LEARNING
# ==============================================================================
with aba_gestor:
    st.subheader(f"📊 Painel Territorial & Desempenho do Modelo Supervisionado — {territorio_sel}")

    k1, k2, k3, k4 = st.columns(4)
    total_urgencia = int((df_base['Classe_Intervencao'] == 0).sum())
    total_alerta = int((df_base['Classe_Intervencao'] == 1).sum())
    k1.metric("Cidadãos Triados", f"{len(df_base):,}".replace(",", "."), "+14% este mês")
    k2.metric("Em Urgência Fisiológica (Classe 0)", f"{total_urgencia}", "Prioridade CRAS/UBS")
    k3.metric("Recall Classe 0 (Random Forest)", f"{metricas_rf['recall_urgencia']*100:.1f}%", "Zero Invisibilidade")
    k4.metric("F1-Score Macro (Teste)", f"{metricas_rf['f1_macro']*100:.1f}%", f"Acc: {metricas_rf['accuracy']*100:.1f}%")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("##### 1. Importância das Variáveis (ML)")
        fig1, ax1 = plt.subplots(figsize=(5, 3.8))
        ax1.barh(df_importancia['Variável'], df_importancia['Importância'], color='#115e59')
        ax1.set_xlabel("Peso Relativo (Gini)")
        plt.tight_layout()
        st.pyplot(fig1)

    with g2:
        st.markdown("##### 2. Matriz de Confusão (Teste)")
        fig2, ax2 = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(
            metricas_rf['cm'], annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['0-Urg.', '1-Alerta', '2-Pronto'],
            yticklabels=['0-Urg.', '1-Alerta', '2-Pronto'], ax=ax2
        )
        ax2.set_xlabel("Classe Prevista")
        ax2.set_ylabel("Classe Real")
        plt.tight_layout()
        st.pyplot(fig2)

    with g3:
        st.markdown("##### 3. Demanda por Zona de Moradia")
        fig3, ax3 = plt.subplots(figsize=(5, 3.8))
        ct = pd.crosstab(df_base['Zona_Moradia'], df_base['Classe_Intervencao'])
        ct.columns = ['0-Urgência', '1-Alerta', '2-Pronto']
        ct.plot(kind='bar', stacked=True, color=['#dc2626', '#d97706', '#059669'], ax=ax3)
        ax3.set_ylabel("Nº de Cidadãos")
        ax3.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig3)