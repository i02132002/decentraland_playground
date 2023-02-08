from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np


class RoadBuilder:
    def __init__(self):
        self.roads = set()

    def add_rect(self, p1, p2):
        for i in range(min(p1[0], p2[0]), max(p1[0], p2[0]) + 1):
            for j in range(min(p1[1], p2[1]), max(p1[1], p2[1]) + 1):
                self.roads.add((i, j))

    def get_roads(self):
        return self.roads


class FeatureGenerator(BaseEstimator, TransformerMixin):

    def __init__(self, isestate=False):
        self.isestate = isestate
        # Read road parcels and genesis plaza parcels
        self.roads = pd.read_csv('roads.csv')
        self.genesis = pd.read_csv('genesis.csv')
        self.district_x = pd.read_csv('district_x.csv')
        self.DU = pd.read_csv('DU.csv')
        self.DC = pd.read_csv('DC.csv')
        self.casino = pd.read_csv('casino.csv')
        main_roads = {}
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
        self.road_parcels = [i for i in zip(self.roads.x, self.roads.y)]
        self.main_road_parcels = list(main_roads.roads)
        self.genesis_parcels = [i for i in zip(self.genesis.x, self.genesis.y)]

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
        feats['dist_road'] = self.calculate_dist(
            self.road_parcels, sold_parcels_xy)
        feats['dist_main_road'] = self.calculate_dist(
            self.main_road_parcels, sold_parcels_xy)
        feats['dist_genesis'] = self.calculate_dist(
            self.genesis_parcels, sold_parcels_xy)
        feats['dist_district_x'] = self.calculate_dist(
            list(zip(self.district_x.x, self.district_x.y)), sold_parcels_xy)
        feats['dist_DU'] = self.calculate_dist(
            list(zip(self.DU.x, self.DU.y)), sold_parcels_xy)
        feats['dist_DC'] = self.calculate_dist(
            list(zip(self.DC.x, self.DC.y)), sold_parcels_xy)
        feats['dist_casino'] = self.calculate_dist(
            list(zip(self.casino.x, self.casino.y)), sold_parcels_xy)

        feats['dist_center'] = np.sqrt(np.sum(np.array(sold_parcels_xy)**2, 1))
        feats['const'] = 1.0
        if self.isestate:
            dist_list = X.groupby(by='id').apply(lambda x: list(zip(x.x, x.y)))
            min_dist_road = dist_list.apply(lambda x: min(calculate_dist(self.road_parcels, x))[
                                            0]).rename('min_dist_road', inplace=True)
            min_dist_genesis = dist_list.apply(lambda x: min(calculate_dist(
                self.genesis_parcels, x))[0]).rename('min_dist_genesis', inplace=True)
            feats = feats.merge(min_dist_road, on='id', how='left')
            feats = feats.merge(min_dist_genesis, on='id', how='left')

        return feats
