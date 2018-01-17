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
    def __init__(self, I, row, col):
        self.row = row
        self.col = col
        self.I_frame = np.mean(np.mean(I[:,row-1:row+1,col-1:col+1], axis=2), axis=1)

    def evaluate(self):
        if self.I_frame.std() > 120:
            return True
        else: 
            return False
                            
class Movie(object):
    def __init__(self, movie_name, data_path):
        self.movie_name = movie_name
        self.movie_path = data_path + '\\' + movie_name   
        movie = Image.open(self.movie_path)
        self.n_frame = movie.n_frames
        self.n_row = movie.size[1]
        self.n_col = movie.size[0]
        
        self.I = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=int)
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
        print(self.n_peaks, 'peaks')
        
    def find_mols(self): # Find intensity changes at peaks
        row = self.peaks[::-1,0]
        col = self.peaks[::-1,1]
        self.mols = []
        for i in range(self.n_peaks):
            mol = Mol(self.I, row[i], col[i])
            if mol.evaluate():
                self.mols.append(mol)    
        print(len(self.mols), 'molecules')  
                        
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
            movie.I_mean = np.mean(movie.I, axis=0)
            movie.I_std = np.std(movie.I, axis=0)
            movie.I_SNR = movie.I_max / movie.I_std
            movie.find_peaks()
            movie.find_mols()
            
                                               
    def plot(self):                  
        plt.close('all')
        # Plot overall movie images
        for i in range(self.movie_num):
            movie = self.movies[i] 
            n_mol = len(movie.mols)
            
            fig1 = plt.figure(self.movie_list[i])    
            sp1 = fig1.add_subplot(121)  
            sp1.imshow(movie.I_max, cmap=cm.gray)
            sp1.set_title('Projected max intensity') 
            
            sp2 = fig1.add_subplot(122) 
            sp2.imshow(movie.I_max, cmap=cm.gray)
            for j in range(n_mol):
                sp2.plot(movie.mols[j].col, movie.mols[j].row, 'ro', markersize=2, alpha=0.5)  
            title2 = 'Molecules = %d' % (n_mol)  
            sp2.set_title(title2)                    
            fig1.tight_layout()

            fig2 = plt.figure()
            sp1 = fig2.add_subplot(221); sp1.hist(movie.I_max.flatten(), bins='scott', histtype='step', color='k'); 
            sp1.set_xscale("log"); sp1.set_yscale("log"); sp1.set_ylabel('Max'); sp1.set_title(np.median(movie.I_max.flatten()))
            
            sp2 = fig2.add_subplot(222); sp2.hist(movie.I_mean.flatten(), bins='scott', histtype='step', color='k'); 
            sp2.set_xscale("log"); sp2.set_yscale("log"); sp2.set_ylabel('Mean'); sp2.set_title(np.median(movie.I_mean.flatten()))
            
            sp3 = fig2.add_subplot(223); sp3.hist(movie.I_std.flatten(), bins='scott', histtype='step', color='k'); 
            sp3.set_xscale("log"); sp3.set_yscale("log"); sp3.set_ylabel('Std'); sp3.set_title(np.median(movie.I_std.flatten()))

            sp4 = fig2.add_subplot(224); sp4.hist(movie.I_SNR.flatten(), bins='scott', histtype='step', color='k'); 
            sp4.set_ylabel('SNR'); sp4.set_title(np.median(movie.I_SNR.flatten()))


            n_fig = 10   
            i_fig = 1                     
            n_col = 10
            n_row = 8
            for j in range(n_mol):
                k = j%(n_col*n_row)
                if k == 0:
                    if i_fig == n_fig:
                        break
                    else:
                        fig_title = "%s %d" % (self.movie_list[i], i_fig)                        
                        fig = plt.figure(fig_title)
                        i_fig += 1
                sp = fig.add_subplot(n_row, n_col, k+1)
                sp.plot(movie.mols[j].I_frame, 'k.', markersize=1)
                title_sp = '%d, %d' % (movie.mols[j].row, movie.mols[j].col)
                sp.set_title(title_sp)
                fig.subplots_adjust(wspace=0.5, hspace=1.0)
                                                                                                                                                      
        plt.show()


              
# Start  
plt.close('all')
data = Data()
data.analysis()
data.plot()