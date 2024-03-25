# -*- coding: utf-8 -*-
"""
AR-RDF Kind
"""

import pandas as pd
import numpy as np

class Leadrdf():
    '''
    
    计算RDF数据建立
    读取数据并处理（添加各个原子的UFF力场下的参数）
    
    '''
    def __init__(self,path_cif_xyz,path_atomval):
        self.path_cif_xyz = path_cif_xyz
        self.path_atomval = path_atomval
    
    def set_data(self):
        data = []
        with open(self.path_cif_xyz, 'r') as file:
            content = file.readlines()
            for i in range(len(content)):
                if i >= 2:
                    temp_list = []
                    for element in content[i].split():#切片分割
                        temp_list.append(element)
                    data.append(temp_list)
        data = pd.DataFrame(data)
        for i in range(1,4):#将数据转为浮点型
            data.iloc[:,i] = pd.to_numeric(data.iloc[:,i],errors='coerce')
        atomVal = self.set_atomVal()
        atom_vallist = self.atom_item(data,atomVal)
        data = data.assign(val=atom_vallist)#并入Dataframe
        return data
    
    def set_atomVal(self):
        atomVal = pd.read_csv(self.path_atomval,header=None)
        return atomVal
    
    def atom_item(self,cif_data,atom_val):
        atom_vallist = []
        vallist = atom_val[1]
        for val in cif_data[0]:
            i=-1
            for atom_val_buff in atom_val[0]:
                i+=1
                if val == atom_val_buff:
                    atom_vallist.append(vallist[i])
        return atom_vallist
    
class Leadgrid():
    '''
    
    读取相关路径
    path_voro_nt2：zeo生成的voro.nt2
    path_atompoint_val: 带有电荷的坐标点.csv格式
    
    grid_type: voro/average
    interation_type: ele/lj
    average_size: et.[6,6,6]
    
    '''
    def __init__(self, grid_type, path_voro_nt2, path_atompoint_val, 
                 path_uff_params, path_cif, average_size=None):
        self.path_voro_nt2 = path_voro_nt2
        self.path_atompoint_val = path_atompoint_val
        self.path_uff_params = path_uff_params
        self.path_cif = path_cif
        self.grid_type = grid_type
        self.average_size = average_size
        
    def set_vorogrid_data(self):
        data = []
        with open(self.path_voro_nt2, 'r') as file:
            content = file.readlines()
            for i in range(len(content)):
                if content[i] != '\n':
                    if i >= 1:
                        temp_list = []
                        for element in content[i].split():
                            temp_list.append(element)
                        data.append(temp_list)
                else:
                    break
            data = pd.DataFrame(data)
            for i in range(1,5):#将数据转为浮点型
                data.iloc[:,i] = pd.to_numeric(data.iloc[:,i],errors='coerce')
        data = data.drop(data[data[4] < 0.8].index)
        return data
    
    def unit_cell(self):
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
        a=b=c=1
        ax = a
        ay = 0.0
        az = 0.0
        bx = b * np.cos(gamma * np.pi / 180.0)
        by = b * np.sin(gamma * np.pi / 180.0)
        bz = 0.0
        cx = c * np.cos(beta * np.pi / 180.0)
        cy = (c * b * np.cos(alpha * np.pi /180.0) - bx * cx) / by
        cz = (c ** 2 - cx ** 2 - cy ** 2) ** 0.5
        unit_cell =  np.asarray([[ax, ay, az],[bx, by, bz], [cx, cy, cz]])
        return length_a,length_b,length_c,unit_cell
    
    def set_avegrid_data(self):
        if self.average_size:
            size1 = self.average_size[0]
            size2 = self.average_size[1]
            size3 = self.average_size[2]
        else:
            size1 = size2 = size3 = 6
        a,b,c,unit_cell = self.unit_cell()
        x = np.linspace(0, a, size1)
        y = np.linspace(0, b, size2)
        z = np.linspace(0, c, size3)
        X,Y,Z = np.meshgrid(x,y,z)
        x_f = X.reshape(1,-1)
        y_f = Y.reshape(1,-1)
        z_f = Z.reshape(1,-1)
        coord_arr = np.concatenate((x_f, y_f, z_f), axis=0)
        coord_arr = np.transpose(coord_arr)
        grid_arr = np.matmul(coord_arr,unit_cell)
        grid_arr = pd.DataFrame(grid_arr)
        grid_arr.insert(0, 'buff', None)#为了符合处理，加一列空列
        return grid_arr
    
    def set_atompoint_data(self):
        data = pd.read_csv(self.path_atompoint_val)
        return data
    
    def set_uff_params(self):
        data = pd.read_csv(self.path_uff_params)
        return data
    
    def set_grid_data(self):
        if self.grid_type == 'voro':
            return self.set_vorogrid_data()
        elif self.grid_type == 'average':
            return self.set_avegrid_data()
        
    
    
    
    
        




































