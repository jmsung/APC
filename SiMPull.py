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
import multiprocessing as mp

def reject_outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def running_avg(x, n):
    return np.convolve(x, np.ones((n,))/n, mode='valid')
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class Mol(object):
    def __init__(self, I, row, col):  
        self.row = row
        self.col = col
        I = np.mean(np.mean(I[:,row-1:row+1,col-1:col+1], axis=2), axis=1)
        I = I - np.min(I)
        BG = np.mean(reject_outliers(I[I < np.max(I)/2]))
        self.I_frame = I - BG
        self.I_max = np.max(self.I_frame)

    def evaluate(self):
        SNR = 7
        n_signal = 7
        
        signal = self.I_frame / self.noise > SNR
        
        if np.sum(signal[n_signal:]*signal[:-n_signal]) > 0:
            return True
        else: 
            return False
            
    def find_noise(self, k):
        n = int(k)
        noise0 = self.I_frame[n:-n]-self.I_avg
        noise1 = reject_outliers(noise0)
        return np.std(noise1)        
                
                                                
class Movie(object):
    def __init__(self, movie_name, data_path):
        self.movie_name = movie_name
        self.movie_path = data_path + '\\' + movie_name   
        movie = Image.open(self.movie_path)
        self.n_frame = movie.n_frames
        self.n_row = movie.size[1]
        self.n_col = movie.size[0]
        self.n_avg = 5
        self.frame = np.arange(self.n_frame)
        self.frame_avg = running_avg(self.frame, self.n_avg)
        
        
        self.I = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=int)
        for i in range(self.n_frame): 
            movie.seek(i) # Move to i-th frame
            self.I[i,] = np.array(movie, dtype=int)
        
    def offset(self):
        I_min = np.min(self.I, axis=0)
        for i in range(self.n_frame):
            self.I[i,] = self.I[i,] - I_min
                        
    def find_peaks(self): # Find local maxima from movie.I_max
        I = self.I_max
        self.peaks = peak_local_max(I, min_distance=2)
        self.n_peaks = len(self.peaks[:, 1])
        print(self.n_peaks, 'peaks')
        
    def find_mols(self): # Find intensity changes at peaks
        row = self.peaks[::-1,0]
        col = self.peaks[::-1,1]
        self.mols = []
        for i in range(self.n_peaks):
            mol = Mol(self.I, row[i], col[i])
            mol.I_avg = running_avg(mol.I_frame, self.n_avg)
            mol.noise = mol.find_noise((self.n_avg-1)/2)
            mol.SNR = mol.I_max / mol.noise
            if mol.evaluate():
                self.mols.append(mol)    
        print(len(self.mols), 'molecules')  
        
    def find_binding(self):
        pass
                                        
                                                                                                                                                                                                
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
            movie.offset() # Pixel-wise bg subtraction, from the minimum intensity of movie 
            movie.I_max = np.max(movie.I, axis=0)
            movie.I_mean = np.mean(movie.I, axis=0)
            movie.I_std = np.std(movie.I, axis=0)
            movie.I_SNR = movie.I_max / movie.I_std
            movie.find_peaks()
            movie.find_mols()
            movie.find_binding()
            
                                               
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


            mol_Imax = []
            mol_noise = []
            mol_SNR = []
            for j in range(n_mol):
                mol_Imax.append(movie.mols[j].I_max)
                mol_noise.append(movie.mols[j].noise)
                mol_SNR.append(movie.mols[j].SNR)            

            fig2 = plt.figure()
            sp1 = fig2.add_subplot(221); sp1.hist(mol_Imax, bins='scott', histtype='step', color='k'); 
            sp1.set_ylabel('Max'); 
            
            sp2 = fig2.add_subplot(222); sp2.hist(mol_noise, bins='scott', histtype='step', color='k'); 
            sp2.set_ylabel('noise'); 

            sp3 = fig2.add_subplot(223); sp3.hist(mol_SNR, bins='scott', histtype='step', color='k'); 
            sp3.set_ylabel('SNR'); 

            n_fig = 10
            i_fig = 1                     
            n_col = 5
            n_row = 4
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
                sp.plot(movie.frame, movie.mols[j].I_frame, 'k', linewidth=1, markersize=3)
                sp.axhline(y=0, color='b', linewidth=1)
                #sp.plot(movie.frame_avg, movie.mols[j].I_avg, 'r', linewidth=3)

                title_sp = '(%d, %d) (noise = %d)' % (movie.mols[j].row, movie.mols[j].col, movie.mols[j].noise)
                sp.set_title(title_sp)
                fig.subplots_adjust(wspace=0.5, hspace=1.0)
                                                                                                                                            
        plt.show()


              
# Start  
plt.close('all')
data = Data()
data.analysis()
data.plot()