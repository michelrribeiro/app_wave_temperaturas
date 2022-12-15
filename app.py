# Data App - Dashboard Interativo Para Avaliação das Temperaturas Máximas Ano a Ano com H2O Wave

# Imports
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scipy.stats as stats
from pmdarima.utils import diff_inv
from scipy.linalg import pinv
from json import load
from statsmodels.tsa.filters.hp_filter import hpfilter

import plotly.graph_objects as go
from plotly import io as pio

from h2o_wave import Q, app, data, main, ui

#############################################################################################
# FUNÇÕES NECESSÁRIAS:
# Carregando dataframe:
def load_data():
    df = pd.read_csv('../data/dados_ano_a_ano.csv', index_col=0)
    return df

# Função para filtrar o dataframe para a localidade específica:
def filter_df(df, country, city, param):
    df = pd.DataFrame(df[(df.Country == country) & (df.City == city)][param], columns=[param])
    df.reset_index(inplace=True)
    df.set_index(pd.to_datetime(df['dt'].map(lambda x: str(x)+'-12-31')), inplace=True)
    df.drop(['dt'], axis=1, inplace=True)
    return df

# Função para retornar os valores diferenciados:
def inv_diff(df_orig, df_diff, periods=1):
    value = np.array(df_orig[:periods].tolist()+df_diff[periods:].tolist())
    inv_diff = diff_inv(value, periods, 1)[periods:]
    return inv_diff

# Função para dividir treino e teste baseado em números de lags:
def size_sample(data, m):
    tam_tot = len(data) - (len(data) % m)
    return int(tam_tot)

# Função de ativação:
def relu(x):
    return np.maximum(x, 0, x)

# Função para as camadas ocultas:
def hidden_nodes(X):
    G = np.dot(X, input_weights)
    G = G + biases
    H = relu(G)
    return H

# Função para predição:
def predict(X):
    out = hidden_nodes(X)
    out = np.dot(out, output_weights)
    return out

# Função com o algoritmo de previsão EML:
def previsoes(df, country, city, param):
    df = filter_df(df, country, city, param)
    df_pred = df.diff(1).dropna()
       
    # Criando o modelo:
    mean_df = np.mean(df_pred)
    sd_df = np.std(df_pred)
    df_pred = pd.DataFrame((df_pred-mean_df)/sd_df)

    m = 15
    size = size_sample(df_pred, m)
    d_train = df_pred[:size]
            
    X_train = np.array([d_train.iloc[i][0] for i in range(m)])
    y_train = np.array(d_train.iloc[m])

    for i in range(1,(d_train.shape[0]-m)):
        l = np.array([d_train.iloc[j][0] for j in range(i,i+m)])
        X_train = np.vstack([X_train,l])
        y_train = np.vstack([y_train, d_train.iloc[i+m]])
            
    input_size = X_train.shape[1]
    hidden_size = 100000
    mu, sigma = 0, 1
    w_lo = -1 
    w_hi = 1
    b_lo = -1 
    b_hi = 1

    #Inicializando valores aleatórios para os pesos e bias:
    global input_weights
    input_weights = stats.truncnorm.rvs((w_lo - mu) / sigma, (w_hi - mu) / sigma, 
                                        loc=mu, scale=sigma,size=[input_size,hidden_size])
    
    global biases
    biases = stats.truncnorm.rvs((b_lo - mu) / sigma, (b_hi - mu) / sigma, 
                                 loc=mu, scale=sigma,size=[hidden_size])
    
    global output_weights         
    output_weights = np.dot(pinv(hidden_nodes(X_train)), y_train)

    # Fazendo as previsões para 50 anos:
    for n in range(50):
        final = len(df_pred)-1
        train = np.array([df_pred.iloc[final-i].values[0] for i in range(m-1, -1, -1)])
        y_pred = predict(train)
        df_pred = df_pred.append(pd.DataFrame({param: y_pred[0]}, 
                                              index=[pd.to_datetime(f'{df_pred.index.max().year+1}-12-31')]))
    df_pred[param] = [round(m[0]*sd_df[0] + mean_df[0], 2) for m in df_pred.values]

    df_inv = pd.DataFrame(np.round(inv_diff(df[param], df_pred[param]), 2), 
                          index=df_pred.index, columns=[param])
    return df_inv

# Gerando t_param até 2013:
def t_param(df, param):
    t_param = (df[df.index < '2014-01-01'].min()[0] if param == 'min' else df[df.index < '2014-01-01'].max()[0])
    return t_param

# Gerando t_param nas previsões:
def t_prev(df, param):
    t_param = (df[df.index > '2013-01-01'].min()[0] if param == 'min' else df[df.index > '2013-01-01'].max()[0])
    return t_param

# Nome a ser utilizado nos cartões:
def card_name(param):
    if param == 'max':
        name = 'Temp. máx.'
    elif param == 'min':
        name = 'Temp. mín.'
    else:
        name = 'Amplitude máx.'
    return name

# Geração da suavização de Hodrick-Prescott:
def hp_decomp(df):
    df = df[df.index < '2014-01-01']
    cycle, trend = hpfilter(df, 6.25)
    df['cycle'] = round(cycle, 2)
    df['trend'] = round(trend, 2)
    return df

# Lista de países:
def countries():
    path = '../data/countries_cities.json'
    mydict = load(open(path))
    countries = list(mydict.keys())
    return countries

# Lista de cidades com base no país:
def cities(country):
    path = '../data/countries_cities.json'
    mydict = load(open(path))
    cities = mydict[country] 
    return cities

# Gerando plots com base em dados definidos:
def plot_serie(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                                x=x, y=y, 
                                mode='lines', 
                                line={'color': 'black'},
                            )
                )

    fig.update_layout(
        title={'text': title, 'x': 0.5}, 
        font=dict(
        family='Times New Roman',
        size=14),
        margin=dict(l=0, r=0, t=25, b=0)
            )

    fig.update_yaxes(visible=False)
    
    return (pio.to_html(fig, include_plotlyjs='cdn', validate=False))

def plot_prev(df, param):
    history = df[df.index < '2014-01-01']
    preds = df[df.index >= '2013-01-01']
            
    fig_prev = go.Figure()
    fig_prev.add_trace(go.Scatter(
                                    name='Dados reais',
                                    x=history.index.year, 
                                    y=history[param], 
                                    mode='lines', 
                                    line={'color': 'black'},
                                )
                        )
    fig_prev.add_trace(go.Scatter(
                                    name='Dados previstos',
                                    x=preds.index.year, 
                                    y=preds[param], 
                                    mode='lines', 
                                    line={'color': 'blue'},
                                )
                        )

    fig_prev.update_layout(
        title={'text': 'Temperaturas reais e previsão para os próximos 50 anos da série', 'x': 0.5}, 
        font=dict(
        family='Times New Roman',
        size=14),
        margin=dict(l=0, r=0, t=25, b=0)
            )


    fig_prev.update_yaxes(visible=False)
    
    return (pio.to_html(fig_prev, include_plotlyjs='cdn', validate=False))

def center_value(value):
    text1 = """<!DOCTYPE html>
                    <html>
                    <style> 
                    .square {display: flex; 
                            align-items: center; 
                            justify-content: center}
                    .square .result{align-self: center; 
                                    font-size: 18px; 
                                    heigth: 10px; 
                                    font-family: 'Arial', serif;}
                    </style>
                    </head>
                    <body>
                    <div class="square"><div class="result">"""
    text2 = """</div>
                </div>
                </body>
                </html>"""
        
    return (text1+f'{value}'+text2)
##############################################################################################################
# DEFININDO O APP:
@app('/temperaturas')
async def serve(q: Q):
    if not q.client.initialized and not q.client.navigator:
        initialize_client(q)
        layout_page(q)

    elif not q.client.navigator:
        q.client.country = q.args.country
        q.client.city = (q.args.city if q.args.city in cities(f'{q.args.country}') else cities(f'{q.args.country}')[0])
        q.client.param = q.args.param
        q.client.navigator = q.args.navigator
        layout_page(q)
        
    else:
        q.client.country = q.args.country
        q.client.city = (q.args.city if q.args.city in cities(f'{q.args.country}') else cities(f'{q.args.country}')[0])
        q.client.param = q.args.param
        q.client.navigator = q.args.navigator
        layout_page(q)
        show_plots(q)
        
    await q.page.save()

##################################################################################################################
# FUNÇÕES DA APLICAÇÃO:
def initialize_client(q: Q) -> None:
    q.client.country = 'Afghanistan'
    q.client.city = 'Baglan'
    q.client.param = 'max'
    q.client.initialized = True

# Definindo layout:
def layout_page(q: Q) -> None:
    #Layout da página:
    q.page['meta'] = ui.meta_card(
        box='',
        layouts=[
            ui.layout(breakpoint='xs', 
            zones=[
                ui.zone(name='header', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('title', size='80%'),
                    ui.zone('navigator', size='20%')
                ]),
                ui.zone(name='selection', direction=ui.ZoneDirection.ROW, zones=[
                    ui.zone('country', size='25%'),
                    ui.zone('city', size='25%'),
                    ui.zone('param', size='16%'),
                    ui.zone('tparam', size='17%'),
                    ui.zone('tprev', size='17%')
                ]),
                ui.zone(name='observed'),
                ui.zone(name='trend'),
                ui.zone(name='content_prev'),
                ui.zone(name='footer')
                ]
            ),
        ]
    )

    ui.zone

    # Título:
    q.page['title'] = ui.header_card(
        box = ui.box(
            'title', 
            height='60px'
            ), 
        icon='Frigid',
        icon_color='White',
        title='Avaliação das temperaturas ao longo dos anos',
        subtitle=''
        )

    # Escolha de visualização:
    q.page['navigator'] = ui.tab_card(
        box=ui.box(
            'navigator', 
            height='60px'
            ),
        value=q.client.navigator,
        link=False,
        name='navigator',
        items=[
            ui.tab(name = 'serie', label ='Série Temporal'), 
            ui.tab(name = 'eml', label = 'Previsão da Série'),
        ],
    )
    # Lista suspensa de países:
    q.page['country'] = ui.form_card(
        box=ui.box('country', size='0'), 
        items=[ui.dropdown(
            name='country', 
            label='Escolha o país', 
            value=q.client.country, 
            required=True, 
            choices=[
                ui.choice(country, country) for country in countries()
                ], 
                trigger=True,
                popup='never',
            ),
        ]
    )

    # Lista suspensa de cidades:
    q.page['city'] = ui.form_card(
        box=ui.box('city'), 
        items=[ui.dropdown(
            name='city', 
            label='Escolha a cidade', 
            value=q.client.city, 
            required=True, 
            choices=[
                ui.choice(city, city) for city in cities(f'{q.client.country}')
                ], 
                trigger=True,
                popup='never',
            ),
        ]
    )

    # Lista suspensa de parâmetros:
    q.page['param'] = ui.form_card(
        box=ui.box('param'), 
        items=[ui.dropdown(
            name='param', 
            label='Escolha o parâmetro', 
            value=q.client.param, 
            required=True, 
            choices=[
                ui.choice('max', 'Temp. Máxima'),
                ui.choice('min', 'Temp. Mínima'),
                ui.choice('max_min', 'Tmáx - Tmín'),
                ], 
                trigger=True,
                popup='never'
            ),
        ]
    )

    # Cartão com temperatura da série real:
    global data_prev
    data_prev = previsoes(
            load_data(),
            country=f'{q.client.country}', 
            city=f'{q.client.city}', 
            param=f'{q.client.param}'
            )

    value_tp = t_param(
                    data_prev, 
                    f'{q.client.param}'
                )

    label_tp = center_value(value_tp)

    q.page['tparam'] = ui.frame_card(
        box=ui.box('tparam'),
        title=f'{card_name(q.client.param)} até 2013',
        content=label_tp
    )

    # Cartão com temperaturas previstas:
    value_tprev = t_prev(
                filter_df(
                    load_data(),
                    country=f'{q.client.country}', 
                    city=f'{q.client.city}', 
                    param=f'{q.client.param}'
                    ), 
                    f'{q.client.param}'
                )

    label_tprev = center_value(value_tprev)

    q.page['tprev'] = ui.frame_card(
        box=ui.box('tprev'),
        title=f'{card_name(q.client.param)} de 2013 a 2063',
        content=label_tprev
    )

    # Fonte de dados:
    q.page['footer'] = ui.footer_card(
        box=ui.box('footer', height='42px'), 
        caption='Dados originais, que vão até 2013, foram obtidos no \
            [Kaggle](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)\
             e então agregados por máxima e mínima no ano.'
        )

def show_plots(q: Q) -> None:
    if q.args.navigator == 'serie':
        del q.page['content_prev']
        
        data = hp_decomp(
                filter_df(
                    load_data(),
                    country=f'{q.client.country}', 
                    city=f'{q.client.city}', 
                    param=f'{q.client.param}'
                    )
                )
    
        fig_obs = plot_serie(
                                x= data.index.year, 
                                y=data[f'{q.client.param}'],
                                title='Temperaturas observadas'
                            )

        fig_trd = plot_serie(
                                x=data.index.year, 
                                y=data['trend'], 
                                title='Tendência das temperaturas com suavização de Hodrick-Prescott'
                            )
    
        q.page['observed'] = ui.frame_card(
            box=ui.box('observed', width='100%', height='200px'),
            title='',
            content=fig_obs
        )

        q.page['trend'] = ui.frame_card(
            box=ui.box('trend', width='100%', height='200px'),
            title='',
            content=fig_trd
        )

    else:
        del q.page['observed']
        del q.page['trend']

        fig_prev = plot_prev(data_prev, param=f'{q.client.param}')

        q.page['content_prev'] = ui.frame_card(
            box=ui.box('content_prev', width='100%', height='400px'),
            title='',
            content=fig_prev
            )