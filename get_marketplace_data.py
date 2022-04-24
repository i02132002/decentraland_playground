from datetime import datetime
import requests
import json
from gql import gql, Client
import pandas as pd
import time
import numpy as np
from sys import argv

from gql.transport.requests import RequestsHTTPTransport

def get_parcels_from_estate(title, owner, limit=None):
    # Select your transport with a defined url endpoint
    transport = RequestsHTTPTransport(url="https://api.thegraph.com/subgraphs/name/decentraland/marketplace")
    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)
    
    mystring = """{{
    estates(first: 1, orderBy: id, where: {{owner: "{0}", id_gt:"{1}"}}) {{
        id
        parcels(first: 1000, orderBy: id, where: {{id_gt:"{2}"}}) {{
          id
          x
          y
        }}
      }}
    }}"""
    
    df = pd.DataFrame()

    #update parameter used in mystring to start querying the database at the earliest update date of sale. The update 
    #date is specified in epoch date and needs to be converted to datetime for human consumption.
    estate_max = 0
    e_count = 0
    i = 0

    while True:
        if limit and i >= limit: break
        parcel_max = 0
        p_count = 0
        dfn = pd.DataFrame()

        query = gql(mystring.format(owner, estate_max, parcel_max))
        result = client.execute(query)
        #if there is no data returned it means you reached the end and should stop querying.
        if len(result['estates']) < 1:
            break

        while True:
            #query the data using GraphQL python library.
            query = gql(mystring.format(owner, estate_max, parcel_max))
            result = client.execute(query)

            if len(result['estates'][0]['parcels']) < 1:
                break
            else:
                #Create a temporary dataframe to later append to the final dataframe that compiles all 1000-row dataframes.
                eid = [result['estates'][0]['id'] for e in result['estates'][0]['parcels']]
                pid = [p['id'] for p in result['estates'][0]['parcels'] ]
                x = [int(p['x']) for p in result['estates'][0]['parcels'] ]
                y = [int(p['y']) for p in result['estates'][0]['parcels'] ]

                dfnn = pd.DataFrame({'eid':eid,'pid':pid,'x':x,'y':y})
                dfn = dfn.append(dfnn, ignore_index=True)

            #Pass into the API the max date from your dataset to fetch the next 1000 records.
            p_count += len(result['estates'][0]['parcels'])
            parcel_max = dfn['pid'].max()

        df = df.append(dfn, ignore_index=True)
        estate_max = df['eid'].max()
        e_count +=1
        print(f'estate {e_count} done, {p_count} parcels found')
        i += 1

    df.to_csv(f'{title}.csv',index=False)
    
if __name__ == '__main__':
    try:
        title = argv[1]
        owner = argv[2]
        if len(argv) == 4:  
            limit = int(argv[2])
            get_parcels_from_estate(title, owner, limit = limit)
        else:
            get_parcels_from_estate(title, owner)
            
    except ValueError:
        print("Stupid user, please enter a number")
        exit(1)
