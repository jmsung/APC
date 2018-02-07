"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from skimage.feature import peak_local_max

def reject_outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def running_avg(x, n):
    return np.convolve(x, np.ones((n,))/n, mode='valid')
   
# Exponential function with cutoff at x = b 
def Exp_cutoff(a, b, x):
    return np.exp(-(x-b)/a)/a * (0.5*(np.sign(x-b)+1))
         
class Mol(object):
    def __init__(self, I, row, col):  
        self.row = row
        self.col = col
        I = np.mean(np.mean(I[:,row-1:row+1,col-1:col+1], axis=2), axis=1)
        I = I - np.min(I)
        BG = np.mean(reject_outliers(I[I < np.max(I)/2]))
        self.I_frame = I - BG
        self.I_max = np.max(self.I_frame)
    
    def find_noise(self):
        self.I_avg = running_avg(self.I_frame, 3)
        noise0 = self.I_frame[1:-1]-self.I_avg
        noise1 = reject_outliers(noise0)
        self.noise = np.std(noise1)    

    def evaluate(self):
        self.SNR_min = 5
        self.SNR_max = 10
        self.dwell_min = 10
        self.dwell_max = 100
        self.blinking = 5
        
        x = running_avg(self.I_frame, 3)
        self.I_s = np.array([x[0]]+x.tolist()+[x[-1]])       
        SNR = self.I_s / self.noise
        signal = SNR > self.SNR_min

        if np.max(SNR) < self.SNR_max:
            return False

        t_b = []
        t_ub = []
        for i in range(len(signal)-1):
            if (signal[i] == False) & (signal[i+1] == True):
                t_b.append(i)
            if (signal[i] == True) & (signal[i+1] == False):
                t_ub.append(i)
        
        if len(t_b)*len(t_ub) == 0: return False 
        if t_ub[0] < t_b[0]: # remove pre-existing binding
            del t_ub[0]
        if len(t_b)*len(t_ub) == 0: return False                
        if t_ub[-1] < t_b[-1]: # remove unfinished binding
            del t_b[-1]
        if len(t_b)*len(t_ub) == 0: return False      

        # combine blinking
        blink_ub = []
        blink_b = []             
        if len(t_b) > 1:  
            for i in range(len(t_b)-1):   
                if abs(t_ub[i] - t_b[i+1]) <= self.blinking: 
                    blink_ub.append(t_ub[i])
                    blink_b.append(t_b[i+1])
                   
            if len(blink_ub) > 0:
                for i in range(len(blink_ub)):
                    t_ub.remove(blink_ub[i])
                    t_b.remove(blink_b[i])

        # delete too short or too long binding
        transient_ub = []
        transient_b = []
        for i in range(len(t_b)):                                      
            if (t_ub[i] - t_b[i] < self.dwell_min) or (t_ub[i] - t_b[i] > self.dwell_max): 
                transient_ub.append(t_ub[i])
                transient_b.append(t_b[i])
                
        if len(transient_b) > 0:
            for i in range(len(transient_b)):
                t_ub.remove(transient_ub[i])
                t_b.remove(transient_b[i])

        if len(t_b)*len(t_ub) == 0: return False    
              
        self.dwell = []     
        self.SNR = []
        self.I_fit = np.zeros(len(signal))          
        for i in range(len(t_b)): 
            self.dwell.append(t_ub[i] - t_b[i])
            I_mean = np.mean(self.I_frame[t_b[i]+1:t_ub[i]-1])
            self.SNR.append(I_mean/self.noise)            
            self.I_fit[t_b[i]+1:t_ub[i]+1] = I_mean
        return True
                                         
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
                        
    # Find local maxima from movie.I_max                    
    def find_peaks(self): 
        I = self.I_max
        self.peaks = peak_local_max(I, min_distance=1)
        self.n_peaks = len(self.peaks[:, 1])
        print(self.n_peaks, 'peaks')
        
    # Find real molecules from the peaks
    def find_mols(self): 
        row = self.peaks[::-1,0]
        col = self.peaks[::-1,1]
        self.mols = []
        self.dwells = []
        self.noise = []
        self.SNR = []
        for i in range(self.n_peaks):
            mol = Mol(self.I, row[i], col[i])
            mol.find_noise()
            if mol.evaluate() is True:
                self.mols.append(mol)    
                self.dwells.extend(mol.dwell)
                self.noise.append(mol.noise)
                self.SNR.extend(mol.SNR)
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
            movie.offset() # Pixel-wise bg subtraction, from the minimum intensity of movie 
            movie.I_max = np.max(movie.I, axis=0)
            movie.I_mean = np.mean(movie.I, axis=0)
            movie.I_std = np.std(movie.I, axis=0)
            movie.I_SNR = movie.I_max / movie.I_std
            movie.find_peaks()
            movie.find_mols()
            movie.dwell_mean = np.mean(movie.dwells)-min(movie.dwells)
            movie.dwell_std = np.std(movie.dwells)
                                                                                                                                                        
    def plot(self):                  
        plt.close('all')
        # Plot overall movie images
        for i in range(self.movie_num):
            movie = self.movies[i] 
            n_mol = len(movie.mols)
            
            # Figure 1
            fig1 = plt.figure(self.movie_list[i])    
            sp1 = fig1.add_subplot(121)  
            sp1.imshow(movie.I_max, cmap=cm.gray)
            sp1.set_title('Projected max intensity') 
            
            sp2 = fig1.add_subplot(122) 
            sp2.imshow(movie.I_max, cmap=cm.gray)
            for j in range(n_mol):
                sp2.plot(movie.mols[j].col, movie.mols[j].row, 'ro', markersize=2, alpha=0.5)  
            title = 'Molecules = %d' % (n_mol)  
            sp2.set_title(title)                    
            fig1.tight_layout()

            # Figure 2
            fig2 = plt.figure(self.movie_list[i]+' Analysis')                
                                    
            sp1 = fig2.add_subplot(121)
            hist_lifetime = sp1.hist(movie.dwells, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
            n_lifetime = len(movie.dwells)*(hist_lifetime[1][1] - hist_lifetime[1][0])
            x_lifetime = np.linspace(0, max(movie.dwells), 1000)

            y_mean = n_lifetime*Exp_cutoff(movie.dwell_mean, min(movie.dwells), x_lifetime) 
            sp1.plot(x_lifetime, y_mean, 'r', linewidth=2)  
            title = 'Mean dwell time = %.1f +/- %.1f (N = %d)' % (movie.dwell_mean, movie.dwell_std/(len(movie.dwells)**0.5), len(movie.dwells))
            sp1.set_title(title)
                        
            sp2 = fig2.add_subplot(122) 
            hist_lifetime = sp2.hist(movie.dwells, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
            sp2.set_yscale('log')
            sp2.semilogy(x_lifetime, y_mean, 'r', linewidth=2)
            title = 'Mean-Std ratio = %.2f' % (movie.dwell_mean / movie.dwell_std)
            sp2.set_title(title)


            # Figure for individual traces  
            n_fig = 1
            i_fig = 1                     
            n_col = 4
            n_row = 5
            for j in range(n_mol):
                k = j%(n_col*n_row)
                if k == 0:
                    if i_fig > n_fig:
                        break
                    else:
                        fig_title = "%s %d" % (self.movie_list[i], i_fig)                        
                        fig = plt.figure(fig_title)
                        i_fig += 1
                sp = fig.add_subplot(n_row, n_col, k+1)
                sp.plot(movie.frame, movie.mols[j].I_frame, 'k.', linewidth=0.5, markersize=3)
                sp.plot(movie.frame, movie.mols[j].I_s, 'b', linewidth=1, markersize=3)
                sp.plot(movie.frame, movie.mols[j].I_fit, 'r', linewidth=2, markersize=3)
                sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)
                sp.axhline(y=movie.mols[j].noise*movie.mols[j].SNR_min, color='k', linestyle='dashed', linewidth=1)

                title_sp = '(%d, %d) (noise = %d)' % (movie.mols[j].row, movie.mols[j].col, movie.mols[j].noise)
                sp.set_title(title_sp)
                fig.subplots_adjust(wspace=0.5, hspace=1.0)
                                                                                                                                            
        plt.show()


              
# Start  
plt.close('all')
data = Data()
data.analysis()
data.plot()