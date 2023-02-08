from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from datetime import timedelta
import yfinance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers

sns.set()


def MANA_to_USD(mana_price, date):
    return float(mana_price[mana_price.reset_index().Date.astype(str) == date].Close)


def convert_to_USD(row):
    return MANA_to_USD(mana_price, row.updatedAt_dt) * row.price


# Getting historical price of MANA in $USD
mana_price = pd.read_csv('MANA-USD.csv')

# Read road parcels and genesis plaza parcels
roads = pd.read_csv('roads.csv')
genesis = pd.read_csv('genesis.csv')
district_x = pd.read_csv('district_x.csv')
DU = pd.read_csv('DU.csv')
DC = pd.read_csv('DC.csv')
casino = pd.read_csv('casino.csv')

# Read historical parcel sales data from parcel_transactions.csv
transactions = pd.read_csv('parcel_transactions.csv')
#transactions = gmd.get_historic_parcels(days=365*5)

# Read parcels currently on sale
on_sale_parcels = pd.read_csv('on_sale_parcels.csv')
on_sale_parcels['estate_size'] = 1

# Read estates currently on sale
#on_sale_estates = pd.read_csv('on_sale_estates.csv')

# All current parcels + estates on sale
#current_on_sale = pd.concat([on_sale_parcels, on_sale_estates])

# Download MANA price data
print('Downloading MANA price from yfinance')
mana_price = yfinance.download('MANA-USD')

# Set rolling window for price normalization
WINDOW = 5

# Calculate the mean MANA price of all parcel transactions
mean_price = (transactions.groupby('updatedAt_dt').mean()
                          .price.to_frame().reset_index())
mean_price['updatedAt_dt'] = mean_price.updatedAt_dt.apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d'))
mean_price = mean_price.set_index('updatedAt_dt')


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


START_DATE = mean_price.index.min()
END_DATE = mean_price.index.max()
all_dates = [sd for sd in daterange(START_DATE, END_DATE)]

# Calculate number of parcels sold for each day
number_sold = transactions.groupby('updatedAt_dt').count().id
number_sold = pd.DataFrame(
    {'Date': number_sold.index, 'number_sold': number_sold.values.astype(int)})
number_sold.Date = number_sold.Date.apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d'))
number_sold = number_sold.set_index('Date')

# Convert sale price to USD for the day of sale
baseline_data = pd.DataFrame(index=all_dates)
baseline_data = baseline_data.join(mean_price, how='left').join(
    number_sold, how='left').join(mana_price.Close, how='left')
baseline_data.rename(columns={'Close': 'mana_price'}, inplace=True)
baseline_data['price_usd'] = baseline_data.price * baseline_data.mana_price
baseline_data.interpolate(method='linear', inplace=True)
baseline_data['price_usd_sm'] = pd.Series.rolling(
    baseline_data.price_usd, window=WINDOW).mean()
baseline_data.price_usd_sm.interpolate(method='backfill', inplace=True)
baseline_data['price_sm'] = pd.Series.rolling(
    baseline_data.price, window=5).mean()
baseline_data.price_sm.interpolate(method='backfill', inplace=True)

cleaned_transactions = transactions[[
    'category', 'price', 'updatedAt', 'updatedAt_dt', 'x', 'y', 'parcel_id']].copy()
cleaned_transactions.rename(columns={'parcel_id': 'id'}, inplace=True)
cleaned_transactions['estate_size'] = 1
cleaned_transactions['total_price'] = cleaned_transactions.price


def MANA_to_USD(row, price_table=mana_price):
    sale_date = row.updatedAt_dt
    return float(price_table.loc[sale_date].Close) * row.iloc[1]


cleaned_transactions['updatedAt_dt'] = cleaned_transactions['updatedAt_dt'].apply(
    lambda x: datetime.strptime(x, '%Y-%m-%d'))
cleaned_transactions['price_usd'] = cleaned_transactions[['updatedAt_dt', 'price']].apply(
    MANA_to_USD, price_table=(mana_price), axis='columns')
cleaned_transactions['total_price_usd'] = cleaned_transactions[[
    'updatedAt_dt', 'total_price']].apply(MANA_to_USD, price_table=(mana_price), axis='columns')
cleaned_transactions = cleaned_transactions.join(
    baseline_data.price_sm, on=cleaned_transactions.updatedAt_dt)
cleaned_transactions.rename(
    columns={'price_sm': 'avg_parcel_price'}, inplace=True)
cleaned_transactions['norm_price'] = cleaned_transactions.price / \
    cleaned_transactions.avg_parcel_price
print('Prepared dataset!')

# Build a feature called main_roads
main_roads = {}


class RoadBuilder:
    def __init__(self):
        self.roads = set()

    def add_rect(self, p1, p2):
        for i in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
            for j in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                self.roads.add((i, j))

    def get_roads(self):
        return self.roads


main_roads = RoadBuilder()
main_roads.add_rect([-11, -11], [12, -10])
main_roads.add_rect([11, -11], [12, 11])
main_roads.add_rect([-11, 10], [12, 11])
main_roads.add_rect([-11, -11], [-10, 11])
main_roads.add_rect([0, 10], [1, 72])
main_roads.add_rect([0, -71], [1, -10])
main_roads.add_rect([-72, -1], [-10, 0])
main_roads.add_rect([11, 0], [71, 1])
main_roads.add_rect([-73, 50], [73, 51])
main_roads.add_rect([-73, -51], [73, -50])
main_roads.add_rect([-51, -73], [-50, 73])
main_roads.add_rect([50, -73], [51, 73])


def calculate_dist(X0, X1):
    '''
    This function takes a list of subject coords of parcels X1 and 
    returns the nearest distance to a list of coords of target parcels X0
    e.g. calculate_dist(road_parcels, sold_parcels)
    '''
    neigh = NearestNeighbors(n_neighbors=1, radius=10.0, algorithm='kd_tree')
    neigh.fit(X0)
    dist, _ = neigh.kneighbors(X1)
    return dist


MAIN_ROAD_PARCELS = list(main_roads.roads)
ROAD_PARCELS = [i for i in zip(roads.x, roads.y)]
GENESIS_PARCELS = [i for i in zip(genesis.x, genesis.y)]


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, isestate=False):
        self.isestate = isestate

    def calculate_dist(self, X0, X1):
        '''
        This function takes a list of subject coords of parcels X1 and 
        returns the nearest distance to a list of coords of target parcels X0
        e.g. calculate_dist(road_parcels, sold_parcels)
        '''
        neigh = NearestNeighbors(
            n_neighbors=1, radius=10.0, algorithm='kd_tree')
        neigh.fit(X0)
        dist, _ = neigh.kneighbors(X1)
        return dist

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = X[['id', 'category', 'updatedAt',
                   'updatedAt_dt', 'x', 'y', 'estate_size']].copy()
        sold_parcels_xy = [i for i in zip(X.x, X.y)]
        feats['dist_road'] = self.calculate_dist(ROAD_PARCELS, sold_parcels_xy)
        feats['dist_main_road'] = self.calculate_dist(
            MAIN_ROAD_PARCELS, sold_parcels_xy)
        feats['dist_genesis'] = self.calculate_dist(
            GENESIS_PARCELS, sold_parcels_xy)
        feats['dist_district_x'] = self.calculate_dist(
            list(zip(district_x.x, district_x.y)), sold_parcels_xy)
        feats['dist_DU'] = self.calculate_dist(
            list(zip(DU.x, DU.y)), sold_parcels_xy)
        feats['dist_DC'] = self.calculate_dist(
            list(zip(DC.x, DC.y)), sold_parcels_xy)
        feats['dist_casino'] = self.calculate_dist(
            list(zip(casino.x, casino.y)), sold_parcels_xy)

        feats['dist_center'] = np.sqrt(np.sum(np.array(sold_parcels_xy)**2, 1))
        feats['const'] = 1.0
        if self.isestate:
            dist_list = X.groupby(by='id').apply(lambda x: list(zip(x.x, x.y)))
            min_dist_road = dist_list.apply(lambda x: min(calculate_dist(ROAD_PARCELS, x))[
                                            0]).rename('min_dist_road', inplace=True)
            min_dist_genesis = dist_list.apply(lambda x: min(calculate_dist(
                GENESIS_PARCELS, x))[0]).rename('min_dist_genesis', inplace=True)
            feats = feats.merge(min_dist_road, on='id', how='left')
            feats = feats.merge(min_dist_genesis, on='id', how='left')

        return feats


fg = FeatureGenerator()

cols = ['x', 'y', 'dist_road', 'dist_main_road', 'dist_center',
        'dist_genesis', 'estate_size', 'dist_district_x',
        'dist_DU', 'dist_DC', 'dist_casino']
cat_columns = ['category']
selector = ColumnTransformer([
    ('selector', 'passthrough', cols),
])

data_augmenter = Pipeline([
    ('transformer', fg),
    ('selector', selector)
])

len_ = len(cleaned_transactions)
X_train, X_test, y_train, y_test = (cleaned_transactions[:round(len_*0.8)],
                                    cleaned_transactions[-round(len_*0.2):],
                                    cleaned_transactions.norm_price[:round(
                                        len_*0.8)],
                                    cleaned_transactions.norm_price[-round(len_*0.2):])


X_train_aug = data_augmenter.fit_transform(X_train)
X_test_aug = data_augmenter.fit_transform(X_test)
X_train_aug, y_train = shuffle(X_train_aug, y_train, random_state=0)
X_test_aug, y_test = shuffle(X_test_aug, y_test, random_state=0)
print('Calculating feature distances...')

nn_normalizer = layers.Normalization(axis=-1)
nn_normalizer.adapt(X_train_aug)

hidden_units1 = 10
hidden_units2 = 10
hidden_units3 = 10

nn_model = tf.keras.Sequential([
    nn_normalizer,
    layers.Dense(hidden_units1, kernel_initializer='normal',
                 activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(hidden_units2, kernel_initializer='normal',
                 activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(hidden_units3, kernel_initializer='normal',
                 activation='relu'),
    layers.Dense(1, kernel_initializer='normal', activation='linear')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='MeanAbsoluteError')

print('Training model...')
history = nn_model.fit(
    X_train_aug,
    y_train,
    epochs=100,
    verbose=2,
    batch_size=64,
    validation_split=0.2)
print('Model trained!')


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

y_train_pred = nn_model.predict(X_train_aug)
print("MAPE: {}%".format(mean_squared_error(y_train, y_train_pred)*100))
print("R2: {}".format(r2_score(y_train, y_train_pred)))

y_test_pred = nn_model.predict(X_test_aug)
print("MAPE: {}%".format(mean_squared_error(y_test, y_test_pred)*100))
print("R2: {}".format(r2_score(y_test, y_test_pred)))

x, y = range(-150, 151), range(-150, 151)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()
R = np.sqrt(X**2 + Y**2)
XY = [i for i in zip(X, Y)]

road_parcels = [i for i in zip(roads.x, roads.y)]
genesis_parcels = [i for i in zip(genesis.x, genesis.y)]
valid_parcels = set(XY) - set(road_parcels) - set(genesis_parcels)
#XY = [i for i in zip(X,Y)]
#X = [i[0] for i in valid_parcels]
#Y = [i[1] for i in valid_parcels]

all_parcels = pd.DataFrame({
    'x': X,
    'y': Y,
    'dist_road': calculate_dist(road_parcels, XY).flatten(),
    'dist_genesis': calculate_dist(genesis_parcels, XY).flatten(),
    'dist_center': np.sqrt(np.sum(np.array(XY)**2, 1)).flatten(),
    'id': None,
    'category': 'parcel',
    'updatedAt': None,
    'updatedAt_dt': None,
    'estate_size': 1
})

print('Calculating predictions for all parcels...')
all_parcels_aug = data_augmenter.fit_transform(all_parcels)
all_parcels['y_pred'] = nn_model.predict(all_parcels_aug)


def delete_row(row):
    if (row.x, row.y) not in valid_parcels:
        row.y_pred = None
    return row


all_parcels = all_parcels.apply(delete_row, axis=1)
all_parcels[['x', 'y', 'y_pred']].to_csv('parcel_model_tf.csv', index=False)
print('Model saved!')
