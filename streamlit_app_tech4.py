import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Configuração da página
st.set_page_config(
    page_title='Pós Tech - Data Analytics, Fase 4',
    layout='wide'
)

# ------------------------------------------------------------------------------------------------------------------------

# Carregando dados
@st.cache_data
def load_data():
    file_path = 'df_diario.csv'
    df = pd.read_csv(file_path)
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna().sort_values(by='ds').reset_index(drop=True)

    # Mais dados
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day_of_week'] = df['ds'].dt.weekday
    df['custom_trend'] = (df['ds'] - df['ds'].min()).dt.days / 365.25
    
    # Criando variáveis sazonais mês e dia da semana, aplicação de one hot encoding
    df = pd.get_dummies(df, columns=['month', 'day_of_week'], drop_first=True)
    
    return df

# ------------------------------------------------------------------------------------------------------------------------

# Carregando os dados
df = load_data()

# ------------------------------------------------------------------------------------------------------------------------

# Título
st.write("Danilo Francisco Pires - RM: 354836")
st.write("Gabriel Martins - RM: 355180")

st.title("Análise e Previsão do Preço do Petróleo")
st.write("- Os dados apresentam grande volatilidade ao longo dos anos, as variações são bem acentuadas e marcadas por eventos globais.")
st.write("- Utilização do modelo Prophet para realização do forecasting")
st.write("- Dados de 20/05/1987 até 04/11/2024")

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Big numbers
min_price = df['y'].min()
max_price = df['y'].max()
mean_price = df['y'].mean()

# Datas dos big numbers
min_price_date = df.loc[df['y'].idxmin(), 'ds'].strftime('%d/%m/%Y')
max_price_date = df.loc[df['y'].idxmax(), 'ds'].strftime('%d/%m/%Y')

# Exibindo big numbers
col1, col2, col3 = st.columns(3)
col1.metric('Valor médio', f"${mean_price:.2f}")
col2.metric('Maior valor', f"${max_price:.2f}")
col2.text(f"Em {max_price_date}")
col3.metric('Menor valor', f"${min_price:.2f}")
col3.text(f"Em {min_price_date}")

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Slider para ajustar o horizonte de previsão
st.subheader("Configuração do Período de Previsão")
dias_previsao = st.slider(
    "Escolha o número de dias para previsão (1 a 180 dias). Tenha em mente que a performance do modelo varia conforme o horizonte de previsão é alterado:",
    min_value=1, max_value=180, value=30, step=1
)

# ------------------------------------------------------------------------------------------------------------------------

st.write("Os gráficos da biblioteca Plotly oferecem uma dinâmica maior aos visuais. Utilize as ferramenta para dar zoom e navegar pelos dados")

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Preparando o prophet
model_data = df[['ds', 'y', 'custom_trend'] + [col for col in df.columns if 'month_' in col or 'day_of_week_' in col]]

# Treinando prophet
model = Prophet(
    daily_seasonality=True,
    yearly_seasonality=True,
    weekly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.1
)

# Variáveis exogenas
model.add_regressor('custom_trend')
for col in model_data.columns:
    if 'month_' in col or 'day_of_week_' in col:
        model.add_regressor(col)

model.fit(model_data)

# ------------------------------------------------------------------------------------------------------------------------

# Horizonte de forecast definido pelo usuário
future = model.make_future_dataframe(periods=dias_previsao)
future['custom_trend'] = (future['ds'] - df['ds'].min()).dt.days / 365 

# ------------------------------------------------------------------------------------------------------------------------

# Dummies
for col in model_data.columns:
    if 'month_' in col or 'day_of_week_' in col:
        future[col] = 0

future['month_' + str(future['ds'].dt.month[0])] = 1
future['day_of_week_' + str(future['ds'].dt.weekday[0])] = 1

# ------------------------------------------------------------------------------------------------------------------------

forecast = model.predict(future)

# ------------------------------------------------------------------------------------------------------------------------

# Gráfico 1: Série Histórica
st.subheader("Série Histórica")
st.write('Comportamento dos preços em USD ao longo do tempo')

fig_historico = go.Figure()
fig_historico.add_trace(go.Scatter(
    x=df['ds'], y=df['y'],
    mode='lines', name='Histórico',
    line=dict(color='blue', width=2)
))

fig_historico.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço do Barril (USD)",
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig_historico)

st.write('Valores muito voláteis e extremamente vulneráveis a eventos em nível global')

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Gráfico 2: Série Histórica + Previsão
st.subheader("Série Histórica + Forecast")
st.write('Adição do forecasting aos dados históricos - Sasonalidade Multiplicativa')

fig_historico_previsao = go.Figure()
fig_historico_previsao.add_trace(go.Scatter(
    x=df['ds'], y=df['y'],
    mode='lines', name='Histórico',
    line=dict(color='blue', width=2)
))
fig_historico_previsao.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'],
    mode='lines', name='Previsão',
    line=dict(color='red', width=2, dash='dash')
))

fig_historico_previsao.update_layout(
    xaxis_title="Data",
    yaxis_title="Preço do Barril (USD)",
    template="plotly_white",
    legend=dict(x=0.01, y=0.99)
)

st.plotly_chart(fig_historico_previsao)

st.write('O modelo prevê uma leve queda nos preços no final de 2024, porém com um pequeno aumento seguindo a série desde a queda de 2015, indicando que o preço, apesar de demonstrar um leve aumente ao longo do ano, andou de lado. A partir de janeiro de 2025, é previsto crescimento conservador.')
st.write('O modelo se mostra bastante sensível às mudanças bruscas da série de dados, o que impacta em sua performance.')

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# tabela de dados

# Textos
st.subheader("Dados Detalhados - Real vs Forecast")
st.write("Use os controles abaixo para filtrar o intervalo de datas e visualizar os valores históricos e previstos.")

# Filtros de data
col_start, col_end = st.columns(2)
with col_start:
    start_date = st.date_input("Data Inicial", value=df['ds'].min().date())
with col_end:
    end_date = st.date_input("Data Final", value=forecast['ds'].max().date())

# Filtro
filtered_data = forecast[(forecast['ds'] >= pd.Timestamp(start_date)) & (forecast['ds'] <= pd.Timestamp(end_date))]
filtered_data = filtered_data.merge(df[['ds', 'y']], on='ds', how='left')
filtered_data = filtered_data[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]

# Format da tabela
filtered_data.columns = ['Data', 'Valor Real (USD)', 'Previsão (USD)', 'Limite Inferior (USD)', 'Limite Superior (USD)']
filtered_data['Data'] = filtered_data['Data'].dt.strftime('%d/%m/%Y')

# Exibindo tabela
st.dataframe(filtered_data)

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Insights
st.subheader("Análises")
st.write("1. Em 1998, o preço do barril de petróleo despencou para cerca de 9 dólares por conta de vários fatores. A crise financeira asiática reduziu a demanda, enquanto o excesso de oferta, impulsionado pelo aumento da produção da OPEP e a retomada das exportações do oriente médio, aumentou o desequilíbrio no mercado. Além disso, a OPEP falhou em controlar a produção para conter a queda. O cenário econômico global agravou o problema, resultando no período de preço mais baixo da história do petróleo.")
st.write("2. Crise de 2008: Um dos pontos mais marcantes na economia global foi a Crise de 2008, onde começou nos EUA com a crise do subprime e do crédito americano. A recessão teve altos impactos para o preço do petróleo No começo, conforme o gráfico mostra no começo de 2008 o preço do petróleo estavam nas máximas, chegando ao pico de USD143,68 decorrente fatores como financiamento geopolítico, alta demanda por parte de economias emergentes e especulação financeira. Porém com o colapso dos bancos sucessivamente (em especial o Lehman Brothers) a lógica do mercado inverteu-se, levando a uma queda acentuada nos preços do petróleo, que despencaram para cerca de US$ 33,73 no fim de 2008. Então, de acordo com os gráficos e com os fatos, podemos inferir que a dinâmica foi de uma alta demanda por petróleo para uso e hedge contra a inflação, pós começo de crise baixa demanda e continuação da oferta global devido a contração do crédito.")
st.write("3. Queda 2014 - 2016: Esse período tem algumas particularidades. Nele não houve guerras, pandemia, crise econômica ou qualquer evento de grande impacto global. Porém, podemos analisar o resultado de um conjunto de fatores: aperfeiçoamento de técnicas na extração do petróleo (Shale Oil - Petróleo de Xisto), altas ofertas no mercado e queda na demanda global (principalmente dos principais players: China e Europa) que deu origem a uma queda vertiginosa até o vale em 2016, quando o barril atingiu o valor de US$ 26,01. Após este período, houve corte na produção pela OPEP (Organização dos Países Exportadores de Petróleo) e outros países não membros (como a Rússia), redução no ritmo de crescimento da produção de petróleo de xisto nos EUA e com isso a recuperação gradual da demanda global.")
st.write("4. Pandemia de COVID-19 (2020): A Pandemia foi mais um evento histórico para a humanidade, porém, diferente de outras crises ela teve grandes impactos na mobilidade urbana demandando isolamento da civilização. Analisando o gráfico e correlacionando com os impactos gerados houve um aquecimento da economia global poucos meses antes do fechamento global tendo aumento de oferta e demanda, em contra partida, conforme os casos foram aumentando a produção continuou nas máximas, porém a demanda diminuiu drasticamente. Isso gerou um desequilíbrio na balança comercial de petróleo, jogando os preços para as mínimas, quando o barril chegou ao menor preço do milênio, de US$ 9,12.")
st.write("5. Alta de 2022-2023: O conflito entre Rússia e Ucrânia também é um ponto de atenção, já que, de acordo com jornal Estadão, a Rússia é uma das principais potências de exportação de petróleo do mundo, responsável por mais de 12% da exportação global de óleo e gás. Vale ressaltar que um de seus principais destinos de exportação é a União Europeia. A guerra entre os 2 países gerou uma crise de abastecimento e demanda global, isto é, a diminuição de um dos maiores exportadores gerou menos produção e aumento de preços. Como o conflito ainda não acabou, os preços ainda não voltaram ao patamar médio da série histórica.")

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Insights
st.subheader("Conclusão")
st.write("Então, podemos concluir que os picos/vales do preço do petróleo é sensível a muitas varáveis, dentre elas podemos citar como as maiores influências: sanções econômicas mundiais, demanda de crédito global, tensões geopolíticas e crises sanitária. De todo modo, esses eventos sempre virão a acontecer, eles não são controláveis portanto e interferem diretamente na prcisão dos modelos de forecasting. O momento do impacto real, a duração e os impactos duradouros são variáveis difíceis de mensurar.")
st.write("O que podemos inferir diante do modelo é que com a expansão econômica, aumento populacional e mudança nos padrões de consumo global a produção de petróleo continuará a ficar em níveis relativamente altos, porém em condições normais reduzindo o preço até chegar em seu platô.")

# ------------------------------------------------------------------------------------------------------------------------

st.divider()

# ------------------------------------------------------------------------------------------------------------------------

# Avaliação da performance do modelo
y_true = model_data['y'][-dias_previsao:].values
y_pred = forecast['yhat'][-dias_previsao:].values

# Cálculo das métricas de avaliação
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Exibindo as métricas
st.subheader("Avaliação da Performance do Modelo")
st.write("Os resultados abaixo variam de acordo com o horizonte de previsão")
st.write('A sasonalidade MULTIPLICATIVA oferece performance melhor em relação a sasonalidade ADITIVA')
st.markdown(f"- Erro Quadrático Médio (MSE): {mse:.2f}")
st.write(f"- Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}")
st.write(f"- Erro Absoluto Médio (MAE): {mae:.2f}")
st.write(f"- Coeficiente de Determinação (R²): {r2:.2f}")
