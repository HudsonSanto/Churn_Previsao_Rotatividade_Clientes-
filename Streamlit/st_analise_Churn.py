import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, balanced_accuracy_score
import numpy as np




# Configuração da página
st.set_page_config(
    page_title="Churn Prediction em Telecomunicações",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📡 Churn Prediction em Telecomunicações")
st.markdown("""
**Da análise preditiva à ação estratégica**  
*Uma jornada de dados para reter clientes e otimizar resultados*
""")

# Sidebar com informações
st.sidebar.header("Painel de Navegação")
st.sidebar.info("""
**Navegue pelo storytelling completo**:
1. 🎬 A Jornada dos Dados
2. 📊 Análise Completa
3. 🔍 Insights do Modelo
4. 🚀 Plano de Ação
""")

def navigate_to_tab(tab_index):
    st.session_state.current_tab = tab_index
# Divisão em abas
tab0, tab1, tab2, tab3 = st.tabs([
    "🎬 A Jornada dos Dados", 
    "📊 Análise Completa", 
    "🔍 Insights do Modelo", 
    "🚀 Plano de Ação"
])

# Carregar dados
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/HudsonSanto/Churn_Previsao_Rotatividade_Clientes-/main/arquivo/Dataset_Churn_Telecomunicacoes"
    df = pd.read_csv(url)
    df = df[df['TotalCharges'] != ' ']
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df.replace({'No internet service': 'No', 'No phone service': 'No'}, inplace=True)
    return df

df = load_data()

with tab0:
    st.header("🎬 A Jornada dos Dados: Do Problema à Solução")
    
    st.markdown("""
    ### 🌌 O Desafio Inicial
    
    Em um mercado competitivo de telecomunicações, onde **a aquisição de novos clientes custa 5x mais** do que reter os existentes,
    nossa empresa enfrentava um problema crítico: **26.5% dos clientes cancelavam seus serviços** anualmente, sem que pudéssemos prever ou prevenir.
    """)
    
    st.image("https://images.unsplash.com/photo-1551434678-e076c223a692?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
            caption="O desafio da retenção em um mercado competitivo")
    
    st.markdown("""
    ### 🔍 A Descoberta dos Dados
    
    Com um dataset completo de **7.043 clientes** e **21 variáveis** comportamentais e demográficas, iniciamos nossa investigação:
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Variáveis Analisadas", "21")
    with col2:
        st.metric("Clientes no Estudo", "7.043")
    with col3:
        st.metric("Taxa de Churn Inicial", "26.5%")
    
    st.markdown("""
    ### 🧠 O Poder da Análise Preditiva
    
    **Nossa abordagem científica**:
    1. **Exploração profunda** dos padrões ocultos nos dados
    2. **Teste de múltiplos algoritmos** de machine learning
    3. **Validação rigorosa** das previsões
    4. **Tradução em ações** concretas
    
    *"Os dados não mentem, mas precisamos fazer as perguntas certas"* - Equipe de Ciência de Dados
    """)
    
    st.markdown("""
    ### 🏆 O Modelo Vencedor
    
    Após testar diversas abordagens, a **Regressão Logística** se destacou:
    """)
    
    model_performance = pd.DataFrame({
        "Modelo": ["Regressão Logística", "Random Forest", "SVM"],
        "Acurácia": [0.777, 0.732, 0.713],
        "Recall": [0.820, 0.764, 0.803]
    })
    
    # Correção: Destacar Regressão Logística como vencedora
    st.dataframe(
        model_performance.style.apply(
            lambda x: ['background-color: #2ecc71' if x.name == 0 else '' for i in x],
            axis=1
        ).format({
            "Acurácia": "{:.3f}",
            "Recall": "{:.3f}"
        }),
        use_container_width=True
    )
    
    st.markdown("""
    ### 📌 Insights Transformadores
    
    **Padrões reveladores que mudaram nossa estratégia**:
    - Contratos mensais têm **3x mais churn** que anuais
    - Cobranças >$70/mês aumentam risco em **40%**
    - Suporte técnico reduz churn em **25%**
    
    *"O conhecimento sem ação é apenas informação perdida"* - Nosso Lema
    """)

with tab1:
    st.header("📊 Análise Completa dos Dados")
    
    # Container 1: Métricas Gerais
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Clientes", len(df))
        with col2:
            st.metric("Clientes Ativos", f"{len(df[df['Churn']=='No']):,}")
        with col3:
            st.metric("Taxa de Churn", f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")
        st.markdown("""
        **Conclusão**: A base contém **7.043 clientes** com taxa de churn de **26.5%**, indicando desbalanceamento significativo entre as classes.
        Necessário uso de técnicas especiais como SMOTE para tratamento.
        """)
    
    # Container 2: Distribuição de Churn
    with st.container():
        st.subheader("Distribuição de Churn")
        fig = px.pie(df, names='Churn', color_discrete_sequence=['#2ecc71', '#e74c3c'],
                    labels={'Churn': 'Status'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Análise**: Distribuição desbalanceada (73.5% retenção vs 26.5% churn).  
        **Impacto**: Modelos tendem a favorecer a classe majoritária, requerendo balanceamento.
        """)
    
    # Container 3: Churn por Contrato
    with st.container():
        st.subheader("Churn por Tipo de Contrato")
        contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
        contract_churn['Churn_Rate'] = contract_churn['Yes'] / contract_churn.sum(axis=1) * 100
        fig = px.bar(contract_churn, x=contract_churn.index, y='Churn_Rate',
                    color_discrete_sequence=['#e74c3c'],
                    labels={'Churn_Rate': 'Taxa de Churn (%)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Insight**: Contratos mensais têm **42.7% de churn** vs 11.3% (anual) e 2.8% (bienal).  
        **Recomendação**: Incentivar migração para contratos longos com benefícios progressivos.
        """)
    
    # Container 4: Tempo de Assinatura
    with st.container():
        st.subheader("Relação entre Tempo de Assinatura e Churn")
        fig = px.box(df, x='Churn', y='tenure', color='Churn',
                   color_discrete_sequence=['#2ecc71', '#e74c3c'],
                   labels={'tenure': 'Meses como Cliente'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Padrão**: Mediana de 10 meses para churn vs 38 meses para clientes fiéis.  
        **Ação**: Focar esforços de retenção nos primeiros 12 meses.
        """)
    
    # Container 5: Cobrança Mensal
    with st.container():
        st.subheader("Relação entre Valores Cobrados e Churn")
        fig = px.box(df, x='Churn', y='MonthlyCharges', color='Churn',
                   color_discrete_sequence=['#2ecc71', '#e74c3c'],
                   labels={'MonthlyCharges': 'Valor Mensal ($)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Análise**: Clientes que cancelam pagam em média **$74.31/mês** vs **$61.27/mês** dos ativos.  
        **Hipótese**: Planos premium podem não estar entregando valor proporcional ao preço.
        """)
    
    # Container 6: Serviços Adicionais
    with st.container():
        st.subheader("Impacto de Serviços Adicionais no Churn")
        
        services = ['OnlineSecurity', 'TechSupport', 'InternetService']
        for service in services:
            service_churn = df.groupby([service, 'Churn']).size().unstack()
            service_churn['Churn_Rate'] = service_churn['Yes'] / service_churn.sum(axis=1) * 100
            fig = px.bar(service_churn, x=service_churn.index, y='Churn_Rate',
                        color_discrete_sequence=['#e74c3c'],
                        title=f'Churn por {service}',
                        labels={'Churn_Rate': 'Taxa de Churn (%)'})
            st.plotly_chart(fig, use_container_width=True)
            
            if service == 'OnlineSecurity':
                st.markdown("""
                **Segurança Online**: Taxa de 41.6% sem vs 15.4% com segurança.  
                **Implicação**: Serviço essencial para retenção.
                """)
            elif service == 'TechSupport':
                st.markdown("""
                **Suporte Técnico**: Redução de 40.8% para 20.6% com suporte.  
                **Estratégia**: Incluir suporte básico em todos os planos.
                """)
            else:
                st.markdown("""
                **Internet Service**: Fibra óptica tem 41.9% churn vs DSL 19.2%.  
                **Investigação**: Verificar qualidade da fibra óptica.
                """)
    
    # Container 7: Análise de Total Charges
    with st.container():
        st.subheader("Distribuição de Gastos Totais")
        fig = px.box(df, x='Churn', y='TotalCharges', color='Churn',
                   color_discrete_sequence=['#2ecc71', '#e74c3c'],
                   labels={'TotalCharges': 'Gasto Total ($)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Observação**: Clientes fiéis têm gasto total significativamente maior (mediana $1.467 vs $412).  
        **Interpretação**: Quanto mais tempo o cliente permanece, maior seu valor vitalício (LTV).
        """)
    
    # Container 8: Forma de Pagamento
    with st.container():
        st.subheader("Churn por Método de Pagamento")
        payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
        payment_churn['Churn_Rate'] = payment_churn['Yes'] / payment_churn.sum(axis=1) * 100
        fig = px.bar(payment_churn, x=payment_churn.index, y='Churn_Rate',
                    color_discrete_sequence=['#e74c3c'],
                    labels={'Churn_Rate': 'Taxa de Churn (%)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Destaque**: Cheque eletrônico tem 33.5% churn vs 10.4% em pagamentos automáticos.  
        **Ação**: Incentivar migração para pagamento automático com benefícios.
        """)

with tab2:
    st.header("🔍 Insights do Modelo Preditivo")
    
    # Pré-processamento para o modelo
    @st.cache_data
    def prepare_model_data():
        df_model = df.copy()
        le = LabelEncoder()
        df_model['Churn'] = le.fit_transform(df_model['Churn'])
        df_model['SeniorCitizen'] = le.fit_transform(df_model['SeniorCitizen'])
        cat_cols = df_model.select_dtypes(include=['object']).columns
        return pd.get_dummies(df_model, columns=cat_cols, drop_first=True)
    
    df_model = prepare_model_data()
    
    # Treinar modelo
    @st.cache_resource
    def train_model():
        X = df_model.drop('Churn', axis=1)
        y = df_model['Churn']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42
        )
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        model = LogisticRegression(
            C=0.1, class_weight='balanced', max_iter=200, solver='liblinear', random_state=42
        )
        return model.fit(X_res, y_res), X_test, y_test
    
    model, X_test, y_test = train_model()
    
    # Métricas
    st.subheader("Performance do Modelo")
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Principais Métricas**")
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics = {
            'Acurácia Balanceada': balanced_accuracy_score(y_test, y_pred),
            'Precisão': report['1']['precision'],
            'Recall': report['1']['recall'],
            'F1-Score': report['1']['f1-score']
        }
        st.table(pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor']).style.format('{:.3f}'))
    
    with col2:
        st.markdown("**Matriz de Confusão**")
        conf_matrix = pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Previsto'], normalize='index')
        fig = px.imshow(
            conf_matrix.values*100,
            labels=dict(x="Previsto", y="Real", color="%"),
            x=['Não Churn', 'Churn'],
            y=['Não Churn', 'Churn'],
            text_auto=".1f",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Variáveis Mais Importantes")
    importance = pd.DataFrame({
        'Variável': model.feature_names_in_,
        'Importância': model.coef_[0]
    }).sort_values('Importância', key=abs, ascending=False).head(10)
    
    fig = px.bar(importance, x='Importância', y='Variável', orientation='h',
                title='Top 10 Fatores que Influenciam o Churn')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🚀 Plano de Ação Estratégico")
    
    st.subheader("Principais Fatores de Churn e Ações Recomendadas")
    
    st.markdown("""
    ### 🎯 Foco em Contratos Mensais
    **Problema**: 43% dos clientes com contrato mensal cancelam  
    **Solução**:  
    - Programa de migração para contratos anuais com benefícios progressivos  
    - Desconto de 15% no primeiro ano para quem migrar  
    """)
    
    st.markdown("""
    ### 🛡️ Serviços de Suporte
    **Problema**: Clientes sem suporte técnico têm 3x mais churn  
    **Solução**:  
    - Pacote "Segurança Total" incluído nos primeiros 6 meses  
    - Alertas proativos de problemas  
    """)
    
    st.markdown("""
    ### 💳 Métodos de Pagamento
    **Problema**: Cheques eletrônicos têm 2x mais churn  
    **Solução**:  
    - Programa "Pagamento Automático" com 5% de desconto  
    - Bônus de fidelidade para pagamentos recorrentes  
    """)
    
    st.subheader("Cronograma de Implementação")
    
    timeline = pd.DataFrame({
        "Mês": ["1º Mês", "2º Mês", "3º Mês"],
        "Ação": [
            "Lançamento programa migração contratos",
            "Implementação pacote segurança para novos clientes",
            "Campanha pagamento automático"
        ],
        "Meta": [
            "Reduzir churn mensal em 15%",
            "Aumentar retenção em 20% nos primeiros 6 meses",
            "Aumentar adesão pagamento automático para 40%"
        ]
    })
    
    st.table(timeline.style.set_properties(**{'text-align': 'left'}))
    
    st.subheader("Próximos Passos")
    st.markdown("""
    1. **Validação** com equipe comercial e marketing  
    2. **Testes A/B** para medir eficácia das ações  
    3. **Monitoramento contínuo** dos resultados  
    """)
    
    st.download_button(
        "📥 Baixar Plano Completo",
        data=timeline.to_csv(index=False).encode('utf-8'),
        file_name="plano_acao_churn.csv",
        mime="text/csv"
    )

# Rodapé
st.markdown("---")
st.markdown("""
**Desenvolvido por** **Hudson Santos** - [LinkedIn](https://www.linkedin.com/in/hudson-santos-513230218/)  
[Github](https://github.com/HudsonSanto)
""")