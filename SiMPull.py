"""
Analysis of SiMPull data (Jongmin Sung)

class Data() 
- path, data_name, load(), movie_list[], movie_num, movies = [Movie()], plot(), analysis()

class Movie() 
- movie_name, movie_path, frame_number, frames = [Frame()], pixels = [Pixel()], molecules = [Molecule()]

class Frame()
- num_frame, frame_intensity 

class Pixel()
- pixel_position, pixel_intensity

class Molecule()
- mol_number, mol_position, mol_intensity 


"""

from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import scipy


class Frame(object):
    def __init__(self, cell_name, sample_path):
        pass

   
class Movie(object):
    def __init__(self, movie_name, data_path):
        self.movie_name = movie_name
        self.movie_path = data_path + '\\' + movie_name   
        movie = Image.open(self.movie_path)
        self.n_frame = movie.n_frames; print(self.n_frame)
        self.row = movie.size[0]; print(self.row)
        self.col = movie.size[1]; print(self.col)
         
        


#        movie_i = Image.open(cell+'-'+ch[0][i]+'.tif')   
#        # Read data from each frame   
#        for j in range(self.n_frame): 
#            movie_i.seek(j) # Move to frame j
#            I0 = np.array(movie_i, dtype=float)
#            self.I[i,j] = I0 - I0.min() 


class Data(object):
    def __init__(self):
        self.data_path = os.getcwd()
        path_split = self.data_path.split('\\')
        self.data_name = path_split[len(path_split)-1]
        self.movie_list = os.listdir(self.data_path) 
        self.movie_num = len(self.movie_list)
        self.movies = [] 
        for movie_name in self.movie_list:
            movie = Movie(movie_name, self.data_path)
            self.movies.append(movie)
                                                
    def analysis(self):
        pass
        
    def plot(self):
        pass
                       



              
# Start  
data = Data()
data.analysis()
data.plot()