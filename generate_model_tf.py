from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
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
from FeatureGenerator import FeatureGenerator
from tensorflow.keras import layers

sns.set()


def MANA_to_USD(mana_price, date):
    return float(mana_price[mana_price.reset_index().Date.astype(str) == date].Close)


def convert_to_USD(row):
    return MANA_to_USD(mana_price, row.updatedAt_dt) * row.price


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def MANA_to_USD(row, price_table):
    sale_date = row.updatedAt_dt
    return float(price_table.loc[sale_date].Close) * row.iloc[1]


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


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_price_map(all_parcels):
    #plt.figure(figsize=(5, 5))
    g = sns.heatmap(np.flip(np.reshape(all_parcels.y_pred.values, (301, 301)),
                    axis=0), vmin=0, vmax=3, square=True, cbar_kws={"shrink": 0.75})
    g.set_facecolor('xkcd:black')
    # plt.axis('equal')
    # plt.axis('off')
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,
        left=False,
        right=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)
    cbar = g.collections[0].colorbar
    ticks_ = np.arange(0, 3, 0.5)
    cbar.set_ticks(ticks_)
    cbar.set_ticklabels(ticks_, size=20)
    plt.show()


def get_all_XY():
    x, y = range(-150, 151), range(-150, 151)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    R = np.sqrt(X**2 + Y**2)
    XY = [i for i in zip(X, Y)]
    return X, Y, XY


def delete_row(row, valid_parcels):
    if (row.x, row.y) not in valid_parcels:
        row.y_pred = None
    return row


def prepare_parcel_dataset():
    # Getting historical price of MANA in $USD
    mana_price = pd.read_csv('MANA-USD.csv')

    # Read historical parcel sales data from parcel_transactions.csv
    transactions = pd.read_csv('parcel_transactions.csv')
    #transactions = gmd.get_historic_parcels(days=365*5)

    # Read parcels currently on sale
    on_sale_parcels = pd.read_csv('on_sale_parcels.csv')
    on_sale_parcels['estate_size'] = 1

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
    cleaned_transactions['updatedAt_dt'] = cleaned_transactions['updatedAt_dt'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d'))
    cleaned_transactions['price_usd'] = cleaned_transactions[['updatedAt_dt', 'price']].apply(
        MANA_to_USD, price_table=mana_price, axis='columns')
    cleaned_transactions['total_price_usd'] = cleaned_transactions[[
        'updatedAt_dt', 'total_price']].apply(MANA_to_USD, price_table=(mana_price), axis='columns')
    cleaned_transactions = cleaned_transactions.join(
        baseline_data.price_sm, on=cleaned_transactions.updatedAt_dt)
    cleaned_transactions.rename(
        columns={'price_sm': 'avg_parcel_price'}, inplace=True)
    cleaned_transactions['norm_price'] = cleaned_transactions.price / \
        cleaned_transactions.avg_parcel_price
    print('Prepared dataset!')
    return cleaned_transactions


def main():
    # Prepare the dataset
    cleaned_transactions = prepare_parcel_dataset()

    # Augment the data with extra features
    cols = ['x', 'y', 'dist_road', 'dist_main_road', 'dist_center',
            'dist_genesis', 'estate_size', 'dist_district_x',
            'dist_DU', 'dist_DC', 'dist_casino']
    cat_columns = ['category']

    selector = ColumnTransformer([
        ('selector', 'passthrough', cols),
    ])
    fg = FeatureGenerator()
    data_augmenter = Pipeline([
        ('transformer', fg),
        ('selector', selector)
    ])

    # Split the data into train/test sets
    len_ = len(cleaned_transactions)
    test_split_fraction = 0.2
    X_train, X_test, y_train, y_test = (cleaned_transactions[:round(len_*(1-test_split_fraction))],
                                        cleaned_transactions[-round(
                                            len_*test_split_fraction):],
                                        cleaned_transactions.norm_price[:round(
                                            len_*(1-test_split_fraction))],
                                        cleaned_transactions.norm_price[-round(len_*test_split_fraction):])

    print('Calculating feature distances...')
    X_train_aug = data_augmenter.fit_transform(X_train)
    X_test_aug = data_augmenter.fit_transform(X_test)
    X_train_aug, y_train = shuffle(X_train_aug, y_train, random_state=0)
    X_test_aug, y_test = shuffle(X_test_aug, y_test, random_state=0)

    # Define the Neural Net
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

    # Train the Neural Net
    print('Training model...')
    history = nn_model.fit(
        X_train_aug,
        y_train,
        epochs=100,
        verbose=2,
        batch_size=64,
        validation_split=0.2)
    print('Model trained!')

    plot_loss(history)

    y_train_pred = nn_model.predict(X_train_aug)
    print("MAPE: {}%".format(mean_squared_error(y_train, y_train_pred)*100))
    print("R2: {}".format(r2_score(y_train, y_train_pred)))

    y_test_pred = nn_model.predict(X_test_aug)
    print("MAPE: {}%".format(mean_squared_error(y_test, y_test_pred)*100))
    print("R2: {}".format(r2_score(y_test, y_test_pred)))

    road_parcels = fg.road_parcels
    genesis_parcels = fg.genesis_parcels
    X, Y, XY = get_all_XY()
    valid_parcels = set(XY) - \
        set(road_parcels) - set(genesis_parcels)

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
    all_parcels = all_parcels.apply(
        lambda x: delete_row(x, valid_parcels), axis=1)

    # Plot the predicted price map
    plot_price_map(all_parcels)
    all_parcels[['x', 'y', 'y_pred']].to_csv(
        'parcel_model_tf.csv', index=False)
    print('Model saved!')


if __name__ == "__main__":
    main()
