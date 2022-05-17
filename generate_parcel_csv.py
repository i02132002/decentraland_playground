from datetime import datetime
import requests
import json
from gql import gql, Client
import pandas as pd
import time
import numpy as np
from sys import argv
import yfinance
from gql.transport.requests import RequestsHTTPTransport
import get_marketplace_data as gmd
import yfinance

gmd.get_parcels_from_estate('roads','0x9a6ebe7e2a7722f8200d0ffb63a1f6406a0d7dce')
gmd.get_parcels_from_estate('genesis','0x4eac6325e1dbf1ac90434d39766e164dca71139e',8)
gmd.get_parcels_from_estate('district_x','0x49907511bdf0351daa417a87d349060876a42544')
gmd.get_parcels_from_estate('DU', '0x71f54536695fc79061bdbf41f8ea54a41cf88108')
gmd.get_parcels_from_estate('DC', '0x6531411be222a663431bde42da60daeb669ccba2')
gmd.get_parcels_from_estate('casino', '0xa65be351527ebcf8c1707d1e444dac38b41a5faf')
gmd.get_all_parcel_transactions()
gmd.get_on_sale_parcels().to_csv('on_sale_parcels.csv',index=False)
yfinance.download('MANA-USD').to_csv('MANA-USD.csv',index=False)
