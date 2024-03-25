# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:28:55 2024

@author: Czl

The database input for Xe/Kr analysis
"""

import numpy as np
import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

class data():
    def __init__(self, 
                 database_name_, 
                 feature_,
                 traget_='Selectivity',
                 feature_list=False,
                 path_rdf = current_dir+'/Feature/AP-RDF/',
                 path_database = current_dir+'/Data/',
                 path_database_nom = current_dir+'/Data_norm/'):
        self.path_database = path_database
        self.database_name = database_name_
        self.feature = feature_
        self.feature_list = feature_list
        self.path_rdf = path_rdf
        self.path_database = path_database
        self.path_database_nom = path_database_nom
        self.traget = traget_
        
    def rdr_in(self, rdf_char=False):
        if self.database_name == 'MOF-2019' or self.database_name == 'MOF-2019-norm':
            result = pd.read_csv(self.path_rdf + 'ap-rdf_2019_epsilon_norm(s).csv')
            return result, result.columns
        if self.database_name == 'MOF-2014' or self.database_name == 'MOF-2014-norm':
            result = pd.read_csv(self.path_rdf + 'ap-rdf_2014_epsilon_norm(s).csv')
            return result, result.columns
        if self.database_name == 'hCOF' or self.database_name == 'hCOF-norm':
            result = pd.read_csv(self.path_rdf + 'ap-rdf_cof_epsilon_norm(s).csv')
            return result, result.columns
        if self.database_name == 'COF7' or self.database_name == 'COF7-norm':
            result = pd.read_csv(self.path_rdf + 'ap-rdf_cof7_epsilon_norm(s).csv')
            return result, result.columns
            
    def data_read(self):
        if self.database_name == 'MOF-2019':
            return pd.read_csv(self.path_database + 'mof_data_7061(s).csv')
        elif self.database_name == 'MOF-2014':
            return pd.read_csv(self.path_database + 'mof_data_2932(s).csv')
        elif self.database_name == 'hCOF':
            return pd.read_csv(self.path_database + 'cof_data_69840(s).csv')
        elif self.database_name == 'COF7':
            return pd.read_csv(self.path_database + 'cof_data_1242(s).csv')
        if self.database_name == 'MOF-2019-norm':
            return pd.read_csv(self.path_database_nom + 'mof_data_7061_norm(s).csv')
        elif self.database_name == 'MOF-2014-norm':
            return pd.read_csv(self.path_database_nom + 'mof_data_2932_norm(s).csv')
        elif self.database_name == 'hCOF-norm':
            return pd.read_csv(self.path_database_nom + 'cof_data_69840_norm(s).csv')
        elif self.database_name == 'COF7-norm':
            return pd.read_csv(self.path_database_nom + 'cof_data_1242_norm(s).csv')

            
    def data_in(self):
        geo = ['Density','ASA','LCD','PLD']
        energy = ['Xe_free-energy','Kr_free-energy','Xe_Henry','Kr_Henry']
        voro = ['Xe_voro','Kr_voro']
        
        data = self.data_read()
        result = self.feature.split("+")
        feature = []
        Chemical = False
        if 'Geometric' in result:
            feature.extend(geo)
        if 'Energy' in result:
            feature.extend(energy)
        if 'Chemical' in result:
            feature.extend(voro)
            Chemical = True
        data_x = data[feature]
        data_y = data[self.traget]
        if Chemical:
            rdf_data, rdf_char = self.rdr_in()
            data_x = np.hstack((data_x,np.array(rdf_data)))
            feature = np.hstack((feature,rdf_char))
        if self.feature_list:
            return np.array(data_x), np.array(data_y), feature
        return np.array(data_x), np.array(data_y)
            
if __name__ == '__main__':
    feature = 'Energy+Chemical'
    
    x_train, y_train, type_ = \
    data('MOF-2019', feature, feature_list=True, traget_='Adsorption_Xe').data_in()








