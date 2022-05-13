from datetime import datetime
from datetime import timedelta
import requests
import json
from gql import gql, Client
import pandas as pd
import time
import numpy as np
from sys import argv


from gql.transport.requests import RequestsHTTPTransport

def get_on_sale_parcels(query_str_filename = 'get_on_sale_parcels.txt'):
    df = get_parcels(query_str_filename)
    df['x'] = df.parcel.apply(lambda a: int(a['x']))
    df['y'] = df.parcel.apply(lambda a: int(a['y']))
    df['parcel_id'] = df.parcel.apply(lambda a: a['id'])
    return df

def get_on_sale_estates(query_str_filename = 'get_on_sale_estates.txt'):
    df = get_parcels(query_str_filename)
    df['estate_id'] = df.estate.apply(lambda a: a['id'])
    df = df.join(df['estate'].apply(pd.Series).parcels)
    df = df[df['parcels'].apply(lambda x: x!=[])]
    each_parcel = df.groupby('estate_id').apply(expand_parcels).set_index('estate_id')
    df = df.set_index('estate_id').join(each_parcel).reset_index(drop=True)
    return df

def get_historic_parcels(query_str_filename = 'get_historic_parcels.txt', days=10):
    return get_parcels(query_str_filename, historic=True, days=days)



def expand_parcels(group):
    row = group.iloc[0]
    return pd.DataFrame({'estate_id': row['estate_id'], 
            'parcel_id': [item['id'] for item in row['parcels']],
            'x': [int(item['x']) for item in row['parcels']],
            'y': [int(item['y']) for item in row['parcels']],
            'estate_size': len(row['parcels'])
            })


def get_parcels(query_str_filename, now=datetime.now().strftime('%s') + '000', historic = False, days=10):
    # Select your transport with a defined url endpoint
    transport = RequestsHTTPTransport(url="https://api.thegraph.com/subgraphs/name/decentraland/marketplace")
    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)
    
    #query_str_filename = 'get_on_sale_parcels.txt'
    with open(query_str_filename) as f:
        query_str = f.read()
    df = pd.DataFrame()

    #update parameter used in mystring to start querying the database at the earliest update date of sale. The update 
    #date is specified in epoch date and needs to be converted to datetime for human consumption.
    update = 1
    if historic:
        now = (datetime.now()  - timedelta(days=days)).strftime('%s')
        update = now

    while True:
        
        #query the data using GraphQL python library.
        query = gql(query_str.format(update, now))
        if historic:
            query = gql(query_str.format(update))
        result = client.execute(query)
        
        #if there is no data returned it means you reached the end and should stop querying.
        if len(client.execute(query)['orders']) <= 1:
            break
    
        else:
            #Create a temporary dataframe to later append to the final dataframe that compiles all 1000-row dataframes.
            df1 = pd.DataFrame()
            df1 = pd.DataFrame(result['orders'])
            #unfold a subdict into a series of columns.
            df1 = df1.join(df1['nft'].apply(pd.Series),lsuffix='_1',rsuffix='_2')     
            
            #append your temp dataframe to your master dataset.
            df = df.append(df1)
            
            #Pass into the API the max date from your dataset to fetch the next 1000 records.
            update = df['updatedAt'].max()
            print("last updated at: {}".format(time.strftime('%Y-%m-%d', time.localtime(int(update)))))

    #reformat the update date in human-readable datetime format.
    df['price'] = df['price'].astype(float)/1e18
    if not historic:
        df['expiresAt'] = df['expiresAt'].astype(float)/1e3
    df['updatedAt_dt'] = df['updatedAt'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(int(x))) )
    return df

    

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
