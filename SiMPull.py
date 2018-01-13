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
from matplotlib import cm
import os
from scipy.optimize import curve_fit
import scipy
from skimage.feature import peak_local_max

class Mol(object):
    def __init__(self, I, col, row):
        self.col = col
        self.row = row
        self.I_frame = I[:,col-1:col+1,row-1:row+1].sum()     
                     
class Movie(object):
    def __init__(self, movie_name, data_path):
        self.movie_name = movie_name
        self.movie_path = data_path + '\\' + movie_name   
        movie = Image.open(self.movie_path)
        self.n_frame = movie.n_frames
        self.n_row = movie.size[0]
        self.n_col = movie.size[1]
        
        self.I = np.zeros((self.n_frame, self.n_col, self.n_row), dtype=int)
        for i in range(self.n_frame): 
            movie.seek(i) # Move to i-th frame
            self.I[i,] = np.array(movie, dtype=int)
        
    def bg(self):
        I_min = np.min(self.I, axis=0)
        for i in range(self.n_frame):
            self.I[i,] = self.I[i,] - I_min
            
    def find_peaks(self): # Find local maxima from movie.I_max
        I = self.I_max
        self.peaks = peak_local_max(I, min_distance=1)
        self.n_peaks = len(self.peaks[:, 1])
        print(self.n_peaks, ' molecules')
        
    def find_mols(self): # Find intensity changes at peaks
        col = self.peaks[:,1]
        row = self.peaks[:,0]
        self.mols = []
        for i in range(self.n_peaks):
            mol = Mol(self.I, col[i], row[i])
            self.mols.append(mol)      
                        
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
            print(movie_name)
                                                
    def analysis(self):
        for i in range(self.movie_num):
            movie = self.movies[i]
            movie.bg() # Pixel-wise bg subtraction, from the minimum intensity of movie 
            movie.I_max = np.max(movie.I, axis=0)
            movie.find_peaks()
            movie.find_mols()
            
                                               
    def plot(self):                  
        plt.close('all')
        # Plot overall movie images
        for i in range(self.movie_num):
            movie = self.movies[i] 
            
            fig1, (sp1, sp2) = plt.subplots(ncols=2)      
            sp1.imshow(movie.I_max, cmap=cm.gray)
#            fig1.colorbar(im1, ax=sp1) 
            sp1.set_title('Projected max intensity')
            
            sp2.imshow(movie.I_max, cmap=cm.gray)
            sp2.plot(movie.peaks[:, 1], movie.peaks[:, 0], 'ro')    
            sp2.set_title('Projected max intensity')            

#            sp2.hist(movie.I_max.flatten(), bins='scott', histtype='step', color='k')  
#            sp2.set_yscale("log") 
#            sp2.set_title('Histogram of max intensity')
                       
            fig1.tight_layout()
                      
                                                                                                                                                           
        plt.show()


              
# Start  
plt.close('all')
data = Data()
data.analysis()
data.plot()