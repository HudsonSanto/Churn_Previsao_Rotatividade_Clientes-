# Churn Prediction em Telecomunicações

Este é um projeto de ciência de dados focado na previsão de rotatividade de clientes (churn) em uma empresa de telecomunicações, utilizando técnicas de machine learning para analisar padrões e prever o comportamento de cancelamento dos clientes.

## 📌 Visão Geral

Este projeto tem como objetivo identificar clientes com maior probabilidade de cancelamento (churn) de serviços de telecomunicações. A antecipação desse comportamento permite que ações preventivas sejam tomadas. Foram utilizados diferentes algoritmos de classificação, destacando-se a Regressão Logística, que obteve o melhor desempenho entre os modelos testados.

## 📊 Principais Resultados

| Modelo              | Acurácia Balanceada | F1-Score | Recall | Precision |
|---------------------|---------------------|----------|--------|-----------|
| **Regressão Logística** | 0.777               | 0.642    | 0.820  | 0.527     |
| **Random Forest**    | 0.732               | 0.589    | 0.764  | 0.479     |
| **SVM**              | 0.713               | 0.564    | 0.803  | 0.435     |

### Insights chave:
- Clientes com contratos mensais têm 3x mais churn do que clientes com contratos anuais.
- Cobranças acima de **$70/mês** aumentam em **40%** o risco de churn.
- Oferecer serviços adicionais, como **suporte técnico** e **segurança**, pode reduzir a taxa de churn.

## 🛠️ Como Executar

1. Clone o repositório:
    ```bash
    git clone https://github.com/seuuser/seurepo.git
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Execute o notebook:
    ```bash
    jupyter notebook Predicao_Churn.ipynb
    ```

## 🔍 Principais Análises

### Distribuição de Churn

A distribuição de churn mostra a taxa de clientes que cancelaram seus serviços em relação aos que permaneceram.

### Fatores que Impactam o Churn
- **Contrato**: Clientes com contratos mensais apresentam maior probabilidade de churn.
- **Tempo de Serviço**: Clientes com menos de 6 meses de contrato são mais propensos ao churn.
- **Serviços Adicionais**: A adição de serviços como segurança e suporte técnico pode reduzir o churn em até **25%**.

## 🧑‍💻 O Dataset

O dataset utilizado neste projeto contém informações sobre clientes de uma empresa de telecomunicações e seus comportamentos em relação ao cancelamento (churn). O conjunto de dados inclui diversas variáveis usadas para prever a probabilidade de churn. Abaixo está uma descrição detalhada das principais colunas presentes no dataset:

| Coluna              | Descrição                                                                 |
|---------------------|---------------------------------------------------------------------------|
| **CustomerID**       | Identificador único para cada cliente.                                    |
| **Gender**           | Gênero do cliente (Male/Female).                                          |
| **SeniorCitizen**    | Indica se o cliente é idoso (1 se sim, 0 se não).                         |
| **Partner**          | Se o cliente tem um parceiro (Yes/No).                                    |
| **Dependents**       | Se o cliente tem dependentes (Yes/No).                                    |
| **Tenure**           | Tempo em meses que o cliente está com a empresa.                          |
| **PhoneService**     | Se o cliente possui serviço de telefonia (Yes/No).                        |
| **MultipleLines**    | Se o cliente tem múltiplas linhas telefônicas (Yes/No).                   |
| **InternetService**  | Tipo de serviço de internet (DSL/Fiber optic/No).                        |
| **OnlineSecurity**   | Se o cliente tem serviço de segurança online (Yes/No).                    |
| **OnlineBackup**     | Se o cliente tem backup online (Yes/No).                                  |
| **DeviceProtection** | Se o cliente tem proteção de dispositivo (Yes/No).                        |
| **TechSupport**      | Se o cliente tem suporte técnico (Yes/No).                                |
| **StreamingTV**      | Se o cliente tem serviço de TV via streaming (Yes/No).                    |
| **StreamingMovies**  | Se o cliente tem serviço de filmes via streaming (Yes/No).                |
| **Contract**         | Tipo de contrato do cliente (Month-to-month/One year/Two year).          |
| **PaperlessBilling** | Se o cliente tem cobrança sem papel (Yes/No).                            |
| **PaymentMethod**    | Método de pagamento (Electronic check/Bank transfer/credit card/Mailed check). |
| **MonthlyCharges**   | Valor da mensalidade que o cliente paga por seus serviços.               |
| **TotalCharges**     | Valor total pago pelo cliente durante seu período com a empresa.         |
| **Churn**            | Se o cliente cancelou o serviço (1 se sim, 0 se não).                    |

### Observações Importantes:
- **Tenure**: É uma das variáveis mais importantes, pois está diretamente relacionada ao tempo de fidelidade do cliente com a empresa. Clientes com maior tenure tendem a ter menor churn.
- **Contract**: Clientes com contratos mensais possuem maior risco de churn em comparação com os clientes com contratos anuais ou bianuais.
- **MonthlyCharges**: Este campo pode ser um bom indicador de churn, visto que valores elevados estão relacionados a uma maior probabilidade de cancelamento.

### Link para o Dashboard no Streamlit:

https://churntelecom.streamlit.app/
