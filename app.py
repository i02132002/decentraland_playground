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



def load_on_sale_parcels():
    return gmd.get_on_sale_parcels()

@st.cache
def load_mana_price():
    mana_price = yfinance.download('MANA-USD')
    return mana_price

st.title('Decentralmate: A virtual real estate price estimator')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load data into the dataframe.
on_sale_parcels = load_on_sale_parcels()
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

st.subheader('Parcels currently for sale')

# Load data into the dataframe.
mana_price = load_mana_price()

#st.subheader('MANA-USD')
#st.write(mana_price)
#st.line_chart(mana_price.Close)

parcel_model = pd.read_csv('parcel_model.csv')
roads = pd.read_csv('roads.csv')

xx, yy = range(-150,151), range(-150,151)
XX,YY = np.meshgrid([x for x in xx], [y for y in yy])
all_p = pd.DataFrame({'x': XX.flatten(), 'y': YY.flatten()})
CURRENT_MANA_PRICE = mana_price.iloc[-1].Close
TEN_DAY_AVG = 15000 / CURRENT_MANA_PRICE #In MANA
all_p = all_p.merge(on_sale_parcels[['x','y','price']], on=['x','y'], how='left')
all_p['price_usd'] = all_p.price * CURRENT_MANA_PRICE
all_p['norm_price'] = all_p.price / TEN_DAY_AVG
all_p = all_p.merge(parcel_model, on=['x','y'], how='left').rename(columns={'y_pred':'norm_price_pred'})
all_p['price_pred_usd'] = all_p.norm_price_pred * TEN_DAY_AVG * CURRENT_MANA_PRICE
all_p['p_ratio'] = all_p.norm_price / all_p.norm_price_pred

def nround(n, m=0):
    try:
        if m == 0:
            return round(n)
        else:
            return round(n, m)
    except (ValueError, TypeError):
        return None
    
hovertext = list(all_p.apply(lambda x: 'x: {}, y: {}<br /> price ($): {}<br />predicted price ($): {}<br />ratio: {}'
                             .format(x.x, x.y, nround(x.price_usd), nround(x.price_pred_usd), nround(x.p_ratio,2)), axis='columns'))

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
    width=600,
    height=600,
    margin=dict(
        l=10,
        r=10,
        b=10,
        t=10,
        pad=10
    ),
)



st.plotly_chart(fig)
st.text_input("Minimum price:", placeholder = '0')
st.text_input("Maximum price:", placeholder = '10,000')
st.button("filter by price", on_click = None)

st.subheader('10 most undervalued parcels')
st.write(all_p.sort_values(by='p_ratio')[:10][['x','y','price_usd','price_pred_usd','p_ratio']])

