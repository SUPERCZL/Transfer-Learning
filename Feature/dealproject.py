# -*- coding: utf-8 -*-
"""

Deal project

"""
import pandas as pd
import numpy as np
import lead
import tqdm as tqdm

class Dealproject():
    def __init__(self,lead_data,type_char):
        if type_char == 'rdf':
            self.cif_data = lead_data.set_data()
            self.grid_data = lead_data.set_data()
            self.num_data = len(lead_data.set_data())
        elif type_char == 'grid':
            self.cif_data = lead_data.set_atompoint_data()
            self.grid_data = lead_data.set_grid_data()
            self.uff_params = lead_data.set_uff_params()
            self.num_data = len(lead_data.set_atompoint_data())
        
    def cal_index_traversal(self,data_1,data_2,char_symbol):#利用np的广播规则建立遍历索引
        if char_symbol == '-':
            data1 = np.array(data_1)
            data2 = np.array(data_2)
            result = data2[:,np.newaxis] - data1
        elif char_symbol == '+':
            data1 = np.array(data_1)
            data2 = np.array(data_2)
            result = data2[:,np.newaxis] + data1
        elif char_symbol == '/':
            data1 = np.array(data_1)
            data2 = np.array(data_2)
            result = data2[:,np.newaxis]/data1
        return result
    
    def set_distance_Index(self, grid_data_buff=None, Split_or_not=None):
        if Split_or_not:
            rij = self.cal_index_traversal(self.cif_data.iloc[:,[1,2,3]], grid_data_buff.iloc[:,[1,2,3]], '-')
        else:
            rij = self.cal_index_traversal(self.cif_data.iloc[:,[1,2,3]], self.grid_data.iloc[:,[1,2,3]], '-')
        rij = np.multiply(rij,rij)
        rij = np.sum(rij, axis=2)
        rij = np.sqrt(rij)
        return rij
    
    def build_array(self,num_data):
        array = np.ones((num_data,num_data))
        array = np.triu(array,k=1)
        return array
    
class ARrdf(Dealproject):
    '''
    
    calculate AP-RDF
    
    '''
    def __init__(self, lead_data, B, r_array):
        super().__init__(lead_data, type_char='rdf')
        self.B = B
        self.r = r_array
        self.num_r = len(r_array)
    
    def set_aij_index(self):
        val = np.array([self.cif_data.val])
        val_shape = np.transpose(val)
        valIndex = np.matmul(val_shape,val)
        return valIndex
    
    def set_e_index(self):
        e_index = []
        rij = self.set_distance_Index()
        for n in range(self.num_r):
            buff1 = (self.r[n]-rij)
            buff2 = np.multiply(buff1,buff1)
            e_index_buff = np.exp(-self.B*buff2)
            e_index.append(e_index_buff)
        return np.array(e_index)
    
    def cal_arrdf_Index(self):
        aij_index = self.set_aij_index()
        e_index = self.set_e_index()
        val_index_index = np.multiply(aij_index,e_index)
        array = self.build_array(self.num_data)
        val_index_index = np.multiply(array,val_index_index)
        one_array = np.ones((1,self.num_data))
        val_index_index = np.matmul(one_array,val_index_index)
        RDF_val = np.matmul(val_index_index,np.transpose(one_array))
        return RDF_val
    
    def rdf_finish(self):
        Val_Index_array = self.cal_arrdf_Index()
        rdf_out_list = []
        for n in range(self.num_r):
            rdf_out = Val_Index_array[n][0,0]
            rdf_out_list.append(rdf_out)
        return rdf_out_list

class Grid(Dealproject):
    '''
    
    calculate Voronoi energy
    interation_type: ele/lj
    
    '''
    def __init__(self, lead_data, interation_type):
        super().__init__(lead_data, type_char='grid')
        self.interation_type = interation_type
        
    def cal_epsilon_ij(self):
        cif_epsilon = np.array([self.cif_data['epsilon']])
        ethonal_epsilon = np.array([self.uff_params['ε']])
        epsilonij = np.dot(np.transpose(ethonal_epsilon),cif_epsilon)
        epsilonij = np.sqrt(epsilonij)
        return epsilonij
    
    def split_data_cal(self, threval):
        num_grid_point = len(self.grid_data)
        n = int(num_grid_point/threval)

        cirnum = 0
        out_buff = 0
        data_index_init = 0
        data_index_next = threval
        while cirnum <= n:
            voro_point_buff = self.grid_data.iloc[data_index_init:data_index_next,[0,1,2,3]]
            data_index_init += threval
            data_index_next += threval
            cirnum += 1

            r_ij = self.set_distance_Index(voro_point_buff, True)
            r_ij = np.where(r_ij >= 12.5, float('inf'), r_ij)
            sigma_ij = self.cal_sigma_ij()
            epsilon_ij = self.cal_epsilon_ij()
            para_index_1 = self.cal_index_traversal(r_ij,sigma_ij,'/')
            para_index = pow(para_index_1,12)-pow(para_index_1,6)
            U_lj_buff = epsilon_ij[:,np.newaxis]*para_index*4
            U_lj_buff = np.sum(U_lj_buff,axis=(1,2))
            #U_lj_buff = np.sum(U_lj_buff,axis=2)
            
            if cirnum == 1:
                out_buff = U_lj_buff
            else:
                out_buff += U_lj_buff

        return out_buff/(num_grid_point)
    
    def cal_sigma_ij(self):
        cif_sigma = self.cif_data['sigema']
        ethonal_sigma = self.uff_params['σ']
        sigmaij = self.cal_index_traversal(cif_sigma, ethonal_sigma,'+')/2
        return sigmaij
        
    def cal_ave_E(self):
        k = 8.99*pow(10,9)
        r_ij = self.set_distance_Index()*pow(10,-10)
        q = np.array([self.uff_params['q']])*1.602176634*pow(10,-38)
        kq = np.transpose(q)*k
        Q = np.array([self.cif_data['charge']])
        kqQ = np.dot(kq,Q)
        E = kqQ[:,np.newaxis]/r_ij
        E = np.sum(E, axis=0)
        E = np.sum(E, axis=1)
        return E
    
    def cal_ave_LJ(self):
        '''
        r_ij = self.set_distance_Index()
        sigma_ij = self.cal_sigma_ij()
        epsilon_ij = self.cal_epsilon_ij()
        para_index_1 = self.cal_index_traversal(r_ij,sigma_ij,'/')
        para_index = pow(para_index_1,12)-pow(para_index_1,6)
        U_lj = epsilon_ij[:,np.newaxis]*para_index*4
        U_lj = np.nan_to_num(U_lj)
        U_lj = np.sum(U_lj,axis=2)
        U_lj = np.mean(U_lj,axis=1)
        #U_lj = np.mean(U_lj, axis=(1,2))
        '''
        threval=4000
        U_lj = self.split_data_cal(threval)
        
        return U_lj
    
    def grid_finish(self):
        if self.interation_type == 'ele':
            return self.cal_ave_E()
        if self.interation_type == 'lj':
            return self.cal_ave_LJ()
                                

class Structure(lead.Leadgrid):
    def __init__(self,path_cif,cutoff = 12.8):
        self.cutoff = cutoff if cutoff else 12.8
        self.path_cif = path_cif
        
    def cal_unit_cell(self):
        with open(self.path_cif, 'r') as cif:
            mof_cif= cif.read()
        for line in mof_cif.split("\n"):
            if "_cell_length_a" in line:
                length_a = line.split()[1]#unit cell vector
                length_a =float(length_a) #string to float
            if "_cell_length_b" in line:
                length_b = line.split()[1]
                length_b = float(length_b)
            if "_cell_length_c" in line:
                length_c= line.split()[1]
                length_c= float(length_c)
            if "_cell_angle_alpha" in line:
                alpha = line.split()[1]
                alpha = float(alpha)
            if "_cell_angle_beta" in line:
                beta= line.split()[1]
                beta= float(beta)
            if "_cell_angle_gamma" in line:
                gamma = line.split()[1]
                gamma = float(gamma)

    #Convert cif information to unit_cell vectors
        ax = length_a
        ay = 0.0
        az = 0.0
        bx = length_b * np.cos(gamma * np.pi / 180.0)
        by = length_b * np.sin(gamma * np.pi / 180.0)
        bz = 0.0
        cx = length_c * np.cos(beta * np.pi / 180.0)
        cy = (length_c * length_b * np.cos(alpha * np.pi /180.0) - bx * cx) / by
        cz = (length_c ** 2 - cx ** 2 - cy ** 2) ** 0.5
        unitcell =  np.asarray([[ax, ay, az],[bx, by, bz], [cx, cy, cz]])
        A = unitcell[0]
        B = unitcell[1]
        C = unitcell[2]
        Wa = np.divide(np.linalg.norm(np.dot(np.cross(B,C),A)), np.linalg.norm(np.cross(B,C)))
        Wb = np.divide(np.linalg.norm(np.dot(np.cross(C,A),B)), np.linalg.norm(np.cross(C,A)))
        Wc = np.divide(np.linalg.norm(np.dot(np.cross(A,B),C)), np.linalg.norm(np.cross(A,B)))
        uc_x = int(np.ceil(self.cutoff/(0.5*Wa)))
        uc_y = int(np.ceil(self.cutoff/(0.5*Wb)))
        uc_z = int(np.ceil(self.cutoff/(0.5*Wc)))
        
        print(uc_x, ' ', uc_y, ' ', uc_z)
        
        return np.array([uc_x, uc_y, uc_z])
        
    
    
    
    
   
        


























