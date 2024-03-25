# -*- coding: utf-8 -*-
"""

Calculate feature values
Author: czl

"""

#import pandas as pd
#import numpy as np
#import cupy as cp

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir+'/Feature/')

import lead
import dealproject
import os

#from tqdm import tqdm

def write_file(output_data, path_out, name):
    output_data.to_csv(path_out + name, index=False,header=0)

if __name__ == '__main__':

    path_cif = current_dir + '/Data/MOF_DATA/mof_cif1.cif'
    path_cif_xyz = current_dir + '/Data/MOF_DATA/mof_cif1.xyz'
    path_atom_val = current_dir + '/Data/mof_LJ_val.csv'
    path_atompoint = current_dir + '/Data/MOF_DATA_potential parameter_iso/mof1_pointcharge.csv'
    path_uff = current_dir + '/Data/Xe-Kr_UFF_val.csv'
    path_voro = current_dir + '/Data/MOF_DATA_voro_r/mof_cif1.nt2'
    path_out = current_dir + '/feature_finish/'

    # calculate ap-rdf
    '''
    set_macheine = lead.Leadrdf(path_cif_xyz, path_atom_val)
    r = np.arange(0,30.25,0.25)
    cal_machine = dealproject.ARrdf(set_macheine, 10, r)
    ar_rdf = cal_machine.rdf_finish()
       
    write_file(out, path_out, 'ap-rdf_.csv')
    '''

    # calculate voronoi energy
    set_macheine = lead.Leadgrid('voro', path_voro, path_atompoint,
                             path_uff, path_cif)
    cal_machine = dealproject.Grid(set_macheine, 'lj')
    grid = cal_machine.grid_finish()

    path_out = '/feature_finish/'
    #write_file(out, path_out, 'voro_.csv')











