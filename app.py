import streamlit as st
import pandas as pd
import numpy as np
import yfinance
from datetime import datetime
import time
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
import get_marketplace_data as gmd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_on_sale_parcels():
    return gmd.get_on_sale_parcels()

@st.cache
def load_mana_price():
    mana_price = yfinance.download('MANA-USD')
    return mana_price

@st.cache(allow_output_mutation=True)
def get_10_day_avg():
    return gmd.get_historic_parcels().price.mean()

st.title('Decentraland Parcel Price Estimator')

# Load data into the dataframe.
on_sale_parcels = load_on_sale_parcels()

# Load data into the dataframe.
mana_price = load_mana_price()


RECENT_MANA = yfinance.Ticker('MANA-USD').history(period='2d', interval='1h')
CURRENT_MANA_PRICE = RECENT_MANA.iloc[-1].Close
PREVIOUS_MANA_PRICE = RECENT_MANA.iloc[-25].Close
PERCENT_CHANGE =  (CURRENT_MANA_PRICE - PREVIOUS_MANA_PRICE) / PREVIOUS_MANA_PRICE *100
ABS_PERCENT_CHANGE = abs(PERCENT_CHANGE)
SIGN = '+' if PERCENT_CHANGE >=0 else '-' 

#up_arrow = Image.open('./resources/up_arrow.png')
#down_arrow = Image.open('./resources/down_arrow.png')



left, right = st.columns(2)
with left:
    st.text('MANA-USD exchange rate:')
with right:
    if SIGN =='+':
        ticker = f'<p style="font-family:Courier; color:Green; font-size: 16px;">1 MANA = ${round(CURRENT_MANA_PRICE, 3)} ({SIGN}{round(ABS_PERCENT_CHANGE,2)}%)</p>'
    else:
        ticker = f'<p style="font-family:Courier; color:Red; font-size: 16px;">1 MANA = ${round(CURRENT_MANA_PRICE, 3)} ({SIGN}{round(ABS_PERCENT_CHANGE,2)}%)</p>'
    st.markdown(ticker, unsafe_allow_html=True)

        
#st.line_chart(mana_price.Close)

parcel_model = pd.read_csv('parcel_model.csv')
roads = pd.read_csv('roads.csv')

xx, yy = range(-150,151), range(-150,151)
XX,YY = np.meshgrid([x for x in xx], [y for y in yy])
all_p = pd.DataFrame({'x': XX.flatten(), 'y': YY.flatten()})

TEN_DAY_AVG = get_10_day_avg() # in USD
#st.text(f'ten day average: {TEN_DAY_AVG}')
all_p = all_p.merge(on_sale_parcels[['x','y','price']], on=['x','y'], how='left')
all_p['price_usd'] = all_p.price * CURRENT_MANA_PRICE
all_p['norm_price'] = all_p.price * CURRENT_MANA_PRICE / TEN_DAY_AVG 
all_p = all_p.merge(parcel_model, on=['x','y'], how='left').rename(columns={'y_pred':'norm_price_pred'})
all_p['price_pred_usd'] = all_p.norm_price_pred * TEN_DAY_AVG
all_p['p_ratio'] = all_p.norm_price / all_p.norm_price_pred

def nround(n, m=0):
    try:
        if m == 0:
            return round(n)
        else:
            return round(n, m)
    except (ValueError, TypeError):
        return None

st.subheader('Parcels currently for sale')


hovertext = ('x: ' + all_p.x.astype(str) + 
            ', y: ' + all_p.y.astype(str) + 
            '<br />price ($): ' + nround(all_p.price_usd).astype(str) +
            '<br />predicted price ($): ' + nround(all_p.price_pred_usd).astype(str) +
            '<br />ratio: ' + nround(all_p.p_ratio,2).astype(str)
)

layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

fig = go.Figure(layout=layout)

fig.add_trace(go.Histogram2d(
    x=roads.x,
    y=roads.y,
    nbinsx=301,
    nbinsy=301,
    hoverinfo = 'skip',
    colorscale = ['#c5c9c7','#87ae73'],
    ))

fig.update_traces(showscale=False)

fig.add_trace(go.Heatmap(
    x=all_p.x,
    y=all_p.y,
    z=all_p.p_ratio,
    colorbar=dict(
                    title='Asking price / predicted price',
                    len=0.75,
                    titleside = 'right',
                    tickfont = dict(size = 20)
                    ),
    hoverinfo='text',
    text=hovertext,
    hoverongaps = False,
    colorscale = 'RdYlBu',
    reversescale = True,
    zmin = 0,
    zmid = 1,
    zmax = 2
))



fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
    visible =False
)

fig.update_xaxes(
    visible =False
)
fig.update_layout(
    width=700,
    height=700,
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0,
        pad=0
    ),
)


st.plotly_chart(fig)
with st.form(key='my_form'):
    filter_by_price = st.checkbox('Filter by price', value = True)
    
    left, right = st.columns(2)
    with left: 
        q_min = st.number_input("Minimum price ($):", min_value=0, max_value=None, value=0)
    with right:
        q_max = st.number_input("Maximum price ($):", min_value=0, max_value=None, value=10000)
    
    filter_by_area = st.checkbox('Filter by area', value = True)

    st.text('Please enter the center coordinate of area you\'d like to query')
    left, mid, right = st.columns(3)
    with left: 
        q_x = st.number_input("x:", value = 0, min_value=-150, max_value=150)
    with mid:
        q_y = st.number_input("y:", value = 0, min_value=-150, max_value=150)
    with right:
        q_r = st.number_input("Query radius:", min_value=1, max_value=None, value = 100)
    
    submitted  = st.form_submit_button(label='Filter by price and/or area')
    
    if submitted:
        st.subheader('Top undervalued parcels')
        cond_nan = ~all_p.price_usd.isna()
        cond_q_min = all_p.price_usd > q_min
        cond_q_max = all_p.price_usd < q_max
        cond_q_r = ((all_p.x - q_x)**2 + (all_p.y - q_y)**2)**0.5 <= q_r
        result = all_p
        if filter_by_area:
            result = all_p[cond_nan & cond_q_r]
        if filter_by_price:
            result = result[cond_nan & cond_q_min & cond_q_max]
        st.write(result.sort_values(by='p_ratio')[:10][['x','y','norm_price','price_usd','price_pred_usd','p_ratio']])


#if st.button("filter by price and area"):
    



