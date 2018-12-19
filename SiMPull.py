"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Single molecule binding and unbinding analysis (Jongmin Sung)

class Data() 
- path, name, load(), list[], n_movie, movies = [Movie()], plot(), analysis(), spot_size, frame_rate, 

class Movie() 
- path, name, n_row, n_col, n_frame, pixel_size, 
- background, spots = [], molecules = [Molecule()]

class Molecule()
- position (row, col), intensity 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import shutil
#from skimage.feature import peak_local_max
#from scipy.optimize import curve_fit
import time
from scipy.optimize import minimize
#from skimage import filters
from skimage.feature import peak_local_max
import platform


frame_rate = 33
n_trace = 50

noise_cutoff = 0.5
spot_size = 3
SNR_min = 10


def reject_outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def running_avg(x, n):
    return np.convolve(x, np.ones((n,))/n, mode='valid')  
   
def Exp(a, x):
    return np.exp(-(x)/a)/a * (0.5*(np.sign(x)+1))  
     
def exp1(x, a, b, c):
    return a + b * np.exp(-x/c)  

def exp2(x, a, b, c, d, e):
    return a + b * np.exp(-x/c) + d* np.exp(-x/e)  

# Exponential function with cutoff at x = b 
def Exp_cutoff(a, b, x):
    return (np.exp(-(x-b)/a)/a) * (0.5*(np.sign(x-b)+1)) + 1e-100

def Exp2_cutoff(a, b, c, d, x):
    return (c*(np.exp(-(x-d)/a)/a) + (1-c)*(np.exp(-(x-d)/b)/b)) * (0.5*(np.sign(x-d)+1)) + 1e-100   

# LogLikelihood 
def LL2(param, d, x):      
    [a, b, c] = param
    return np.sum(np.log10(Exp2_cutoff(a, b, c, d, x)))  

def MLE2(a, b, c, d, x): 
    fun = lambda *args: -LL2(*args)
    p0 = [a, b, c]
    result = minimize(fun, p0, method='SLSQP', args=(d, x)) 
    print(result)
    return result

def LL1(param, b, x):      
    [a] = param
    return np.sum(np.log10(Exp_cutoff(a, b, x)))  

def MLE1(a, b, x): 
    fun = lambda *args: -LL1(*args)
    p0 = [a]
    result = minimize(fun, p0, method='SLSQP', args=(b, x)) 
#    print(result)
    return result

        
class Mol:
    def __init__(self, I, row, col):  
        self.row = row
        self.col = col
        s = int(spot_size/2)
        self.I_frame = np.mean(np.mean(I[:,row-s:row+s,col-s:col+s], axis=2), axis=1)
        
    def normalize(self):
        I = self.I_frame     
        I = I - np.min(I)
        I = I/np.max(I)

        bg_u = np.mean(I[I < 0.4])
        bg_b = np.mean(I[I > 0.6])
        self.I_frame = (I - bg_u)/(bg_b - bg_u) 
               
    def find_noise(self):
        self.I_avg = running_avg(self.I_frame, 3)
        noise0 = self.I_frame[1:-1]-self.I_avg
        noise1 = reject_outliers(noise0)
        self.noise = np.std(noise1)    

    def evaluate(self):#, SNR_min, dwell_min, dwell_max):
        blinking = 1
        dwell_min = 1
        dwell_max = 100

        if self.noise > 1/SNR_min: return False        
        
        x = running_avg(self.I_frame, 3)
        self.I_s = np.array([x[0]]+x.tolist()+[x[-1]])       
        signal = self.I_s > noise_cutoff

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
                if abs(t_ub[i] - t_b[i+1]) <= blinking: 
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
            if (t_ub[i] - t_b[i] < dwell_min): 
                transient_ub.append(t_ub[i])
                transient_b.append(t_b[i])

            if (t_ub[i] - t_b[i] > dwell_max): 
                transient_ub.append(t_ub[i])
                transient_b.append(t_b[i])
                
        if len(transient_b) > 0:
            for i in range(len(transient_b)):
                t_ub.remove(transient_ub[i])
                t_b.remove(transient_b[i])

        if len(t_b)*len(t_ub) == 0: return False    
              
        self.dwell = []  
        self.waiting = []   
        self.SNR = []
        self.I_fit = np.zeros(len(signal))          
        for i in range(len(t_b)): 
            self.dwell.append(t_ub[i] - t_b[i])
            if i < len(t_b)-1:
                self.waiting.append(t_b[i+1] - t_ub[i])
            I_mean = np.mean(self.I_frame[t_b[i]:t_ub[i]])
            self.SNR.append(I_mean/self.noise)            
            self.I_fit[t_b[i]+1:t_ub[i]+1] = I_mean
        return True
                                         
class Movie:
    def __init__(self):
        pass
        
    def read(self, folder_path, movie_name):
        self.folder_path = folder_path        
        self.movie_name = movie_name
        print('Movie = ', movie_name)
        self.movie_path = os.path.join(folder_path, movie_name) 
        movie = Image.open(self.movie_path)
        self.n_frame = movie.n_frames
        self.n_row = movie.size[1]
        self.n_col = movie.size[0]
        print('[frame, row, col] = [%d, %d, %d] \n' %(self.n_frame, self.n_row, self.n_col))        
          
        self.I = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=int)
        for i in range(self.n_frame): 
            movie.seek(i) # Move to i-th frame
            self.I[i,] = np.array(movie, dtype=int)
            
    def crop(self, I, x, y, s):
        hs = int(s/2)
        I0 = I[x-hs:x+hs+1, y-hs:y+hs+1]
        val = skimage.filters.threshold_otsu(I0)
        mask = I0 > val
        return mask
        
                      
    def drift(self):
        r = 20  
        size = min(self.n_row, self.n_row) - (2*r+10)

        I0 = self.I[0,] # 0th frame 
        cx = int(self.n_row/2)
        cy = int(self.n_col/2) 
        I0s = self.crop(I0, cx, cy, size)     
        self.I0s = I0s
     
        self.drift = np.zeros((self.n_frame, 2*r+1, 2*r+1), dtype=float)  
     
        self.drift_x = []
        self.drift_y = []

        for i in range(self.n_frame):  
            print(i)
            I1 = self.I[i,] # ith frame  
            for j in range(-r, r+1):
                for k in range(-r, r+1):
                    I1s = self.crop(I1, cx+j, cy+k, size)
                    corr = np.sum(I0s*I1s)
                    self.drift[i, j+r, k+r] = corr
                                        
            self.drift[i,] = self.drift[i,] - self.drift[i,].min()
            self.drift[i,] = self.drift[i,]/self.drift[i,].max()

            dr = np.argwhere(self.drift[i,] == 1)
            self.drift_x += [dr[0][1]-r]
            self.drift_y += [dr[0][0]-r]

        
    def drift_correct(self):
        I = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=int)
        dx = self.drift_x
        dy = self.drift_y
        
        for i in range(self.n_frame):  
            for j in range(self.n_row):
                if j-dy[i] >= 0 and j-dy[i] < self.n_row: 
                    for k in range(self.n_col):
                        if k-dx[i] >= 0 and k-dx[i] < self.n_col:                    
                            I[i, j-dy[i], k-dx[i]] = self.I[i, j, k]                                  
        self.I = I
                   
    def offset(self):
        I_min = np.min(self.I, axis=0)
        for i in range(self.n_frame):
            self.I[i,] = self.I[i,] - I_min

    # Find local maxima from movie.I_max                    
    def find_peaks(self):#, spot_size): 
        I = self.I_max
#        self.peaks = skimage.feature.peak_local_max(I, min_distance=int(spot_size*1.5))
        self.peaks = peak_local_max(I, min_distance=int(spot_size*1.5))        
        self.n_peaks = len(self.peaks[:, 1])
        print('\nFound', self.n_peaks, 'peaks. ')
        
    # Find real molecules from the peaks
    def find_mols(self):#, spot_size, SNR_min):#, dwell_min, dwell_max): 
        row = self.peaks[::-1,0]
        col = self.peaks[::-1,1]
        self.mols = []
        self.dwells = []
        self.noise = []
        self.SNR = []
        for i in range(self.n_peaks):
            mol = Mol(self.I, row[i], col[i])#, spot_size)
            mol.normalize()
            mol.find_noise()
#            if mol.evaluate(SNR_min, dwell_min, dwell_max) is True:
            if mol.evaluate() is True:
                self.mols.append(mol)    
                self.dwells.extend(mol.dwell)
                self.noise.append(mol.noise)
                self.SNR.extend(mol.SNR)
        print('Found', len(self.mols), 'molecules. \n')  
     
    def find_dwelltime(self):
        x = np.array(self.dwells)
        result1 = MLE1(np.mean(self.dwells), np.min(self.dwells), x)
        self.dwell_fit1 = result1["x"]
#        result2 = MLE2(np.mean(self.dwells)/2, np.mean(self.dwells)*2, 0.5, np.min(self.dwells), x)
#        self.dwell_fit2 = result2["x"]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    def correlation(self, spot_size, n_corr):
        n_frame = self.n_frame
        n_peak = self.n_peaks
        row = self.peaks[::-1,0]
        col = self.peaks[::-1,1]

        self.I_peak = np.zeros((n_peak, n_frame))   
        self.corr = np.zeros((n_peak, n_corr))     

        print("Analyzing correlation. ")
        t0 = time.clock()

        for i in range(n_peak):
            s = int(spot_size/2)
            I = np.mean(np.mean(self.I[:,row[i]-s:row[i]+s,col[i]-s:col[i]+s], axis=2), axis=1)
            
            I = I - np.min(I)
            I = I/np.max(I)

            bg_u = np.mean(I[I < 0.5])
            bg_b = np.mean(I[I > 0.5])
            self.I_peak[i] = (I - bg_u)/(bg_b - bg_u) 

            for j in np.arange(1, n_corr+1):
                corr = []
                for k in range(n_frame-j):
                    corr.append(self.I_peak[i][k]*self.I_peak[i][k+j])
                self.corr[i, j-1] = np.mean(corr)

            if i > 0 and (i % int(n_peak/10) == 0) :
                t1 = time.clock()                  
                print("%d %% done. %.1f min passed. " %(int(i/n_peak*100+1), (t1-t0)/60))

        self.corr_mean = np.mean(self.corr, axis=0)
        self.corr_sem = np.std(self.corr, axis=0)/n_peak**0.5
        self.corr_var = self.corr_sem**2

    def plot_image(self, path):    
        # Figure 1 - Image                                                    
        fig1 = plt.figure(1, figsize = (20, 10), dpi=300)    
        
        sp1 = fig1.add_subplot(131)  
        sp1.imshow(self.I_min, cmap=cm.gray)
        sp1.set_title('Minimum intensity')
        
        sp2 = fig1.add_subplot(132)  
        sp2.imshow(self.I_max, cmap=cm.gray)
        sp2.set_title('Maximum intensity') 
        
        sp3 = fig1.add_subplot(133) 
        sp3.imshow(self.I_max, cmap=cm.gray)
        for j in range(len(self.mols)):
            sp3.plot(self.mols[j].col, self.mols[j].row, 'ro', ms=2, alpha=0.5)  
        title = 'Molecules = %d, Spot size = %d' % (len(self.mols), spot_size)  
        sp3.set_title(title)      
        
        fig1.savefig(os.path.join(path,'Image.png'))   
        plt.close(fig1)

    def plot_histogram(self, path):    
        # Figure 2 - Histogram: Single Exp           
        fig2 = plt.figure(2, figsize = (20, 10), dpi=300)  
        
        sp1 = fig2.add_subplot(121)  
#        if self.auto_bin == 'n':
#            bins = np.arange(self.dwell_min, max(self.movie.dwells), self.bin_size)   
#        else:            
#            bins = 'scott'          
                                   
        hist_lifetime = sp1.hist(self.dwells, bins ='scott' , normed=False, color='k', histtype='step', linewidth=2)
  
        n_lifetime = len(self.dwells)*(hist_lifetime[1][1] - hist_lifetime[1][0])
        x_lifetime = np.linspace(0, max(self.dwells), 1000)
        y_mean = n_lifetime*Exp_cutoff(self.dwell_mean, 1, x_lifetime) 
        y_fit = n_lifetime*Exp_cutoff(self.dwell_fit1[0], 1, x_lifetime)
        sp1.plot(x_lifetime, y_mean, 'b', x_lifetime, y_fit, 'r', linewidth=2)  
        title = '\nMean dwell time [s] = %.2f +/- %.2f (N = %d)' % (frame_rate*self.dwell_mean, frame_rate*self.dwell_std/(len(self.dwells)**0.5), len(self.dwells))
        sp1.set_title(title)
        print(title)
             
        sp2 = fig2.add_subplot(122) 
        hist_lifetime = sp2.hist(self.dwells, bins='scott', normed=False, color='k', histtype='step', linewidth=2)
        sp2.set_yscale('log')
        sp2.semilogy(x_lifetime, y_mean, 'b', x_lifetime, y_fit, 'r', linewidth=2)
        sp2.axis([0, 1.1*max(x_lifetime), 0.5, 2*max(y_mean)])
        title = 'Mean dwell time [frame] = %.1f +/- %.1f (N = %d)' % (self.dwell_mean, self.dwell_std/(len(self.dwells)**0.5), len(self.dwells))
        sp2.set_title(title)

        fig2.savefig(os.path.join(path,'Histogram.png'))
        plt.close(fig2)
                  
    def plot_traces(self, path):                                                                                                                                                                                                                                                                  
#        self.n_avg = 5
#        self.frame = np.arange(self.n_frame)
#        self.frame_avg = running_avg(self.frame, self.n_avg)

        # Figure for individual traces                              
        fig_path = os.path.join(path, "Traces")
        if os.path.exists(fig_path):
            shutil.rmtree(fig_path)
            os.makedirs(fig_path)
        else:
            os.makedirs(fig_path)
                
        frame = np.arange(self.n_frame)
                
        n_fig = min(n_trace, len(self.mols))        
        for j in range(n_fig):                 
            fig = plt.figure(100, figsize = (25, 15), dpi=300)
            sp = fig.add_subplot(111)
            sp.plot(frame, self.mols[j].I_frame, 'k', lw=0.5, ms=3)
#            sp.plot(frame, self.mols[j].I_s, 'b', linewidth=1, markersize=3)
            sp.plot(frame, self.mols[j].I_fit, 'r', lw=1, ms=3)
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)
            sp.axhline(y=noise_cutoff, color='k', linestyle='dotted', lw=1)
            sp.axhline(y=1, color='k', linestyle='dotted', lw=1)                
            title_sp = '(%d, %d) (noise = %.2f)' % (self.mols[j].row, self.mols[j].col, self.mols[j].noise)
            sp.set_title(title_sp)
            fig.subplots_adjust(wspace=0.3, hspace=0.5)
            print("Save Trace %d (%d %%)" % (j+1, ((j+1)/n_fig)*100))
            fig.savefig(os.path.join(fig_path,'Trace%d.png') % (j+1)) 
            fig.clf()
                    
    def plot_drift(self, path):
        fig8 = plt.figure(1, figsize = (20, 10), dpi=300)    
        fig8.clf()
        movie = self.movie
                
        sp1 = fig8.add_subplot(221)  
        im1 = sp1.imshow(movie.I0s, cmap=cm.gray)
        plt.colorbar(im1)
        sp1.set_title('Kernel (Frame = 0)')

        frame = movie.n_frame -1
        sp2 = fig8.add_subplot(222)  
        r = len(movie.drift[0])
        im2 = sp2.imshow(movie.drift[frame], cmap=cm.gray, extent = [-r, r, r, -r])
        plt.colorbar(im2)
        sp2.set_title('Correlation (Frame = %d)' %(frame))

        sp3 = fig8.add_subplot(223)
        sp3.plot(movie.drift_x, 'k')
        sp3.set_title('Drift in X')  

        sp4 = fig8.add_subplot(224)
        sp4.plot(movie.drift_y, 'k')
        sp4.set_title('Drift in Y') 

        fig8.savefig(os.path.join(path, 'Fig8_Drift.png'))   
        plt.close(fig8)
            
                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
class Data:
    def __init__(self):
        pass
        
    def read(self):
        data_path = os.getcwd()

        if platform.system() == 'Windows':
            path_split = data_path.split('\\')    
        else:
            path_split = data_path.split('/')   
            
        data_name = path_split[len(path_split)-1]      
        print('Data = %s \n' %(data_name))
        
        folder_list = os.listdir(data_path)                      
        print('%d movies are found. \n' %(len(folder_list)))

        self.movies = []
        for i in range(len(folder_list)):
            folder_path = os.path.join(data_path, folder_list[i]) 
            file_list = os.listdir(folder_path)
            for j in range(len(file_list)):
                if file_list[j][-4:] == '.tif':
                    movie = Movie()
                    movie.read(folder_path, file_list[j])
                    self.movies.append(movie)            
                                    
        #self.frame_rate = float(input('How many frames per second? '))
        #self.n_corr = int(input('How many frames for correlation [100]? '))
        #self.spot_size = int(input('Spot size in pixel [3]? '))
        #self.SNR_min = int(input('Signal to noise cutoff [10]? '))

        
#        self.auto_dwell = input('Automatic range for dwell time [y/n]? ')
#        if self.auto_dwell == 'n':         
#            self.dwell_min = float(input('Minimum dwell time [s]? ')) 
#            self.dwell_max = float(input('Maximum dwell time [s]? ')) 
            
#        self.auto_wait = input('Automatic range for wait time [y/n]? ')
#        if self.auto_wait == 'n':         
#            self.wait_min = float(input('Minimum wait time [s]? ')) 
#            self.wait_max = float(input('Maximum wait time [s]? ')) 
                                             
#        self.auto_bin = input('Automatic bin size [y/n]? ')
#        if self.auto_bin == 'n':
#            self.bin_size = float(input('Bin size? ')) 
            
#        self.drift_corr = input('Drift correction [y/n]? ')
                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    def analyze(self):
        for i in range(len(self.movies)):
            movie = self.movies[i]            
            print('Analyzing... ',  movie.movie_name)

#            if self.drift_corr == 'y':
#                movie.drift()
#                movie.drift_correct()
            
            movie.I_min = np.min(movie.I, axis=0)
            movie.offset() # Pixel-wise bg subtraction, from the minimum intensity of movie 
            movie.I_max = np.max(movie.I, axis=0)
            movie.find_peaks()#spot_size)
            movie.find_mols()#spot_size, SNR_min)#, self.dwell_min, self.dwell_max)
            movie.find_dwelltime()
            movie.dwell_mean = np.mean(movie.dwells)-min(movie.dwells)
            movie.dwell_std = np.std(movie.dwells)
    #        movie.correlation(self.spot_size, self.n_corr)
                                                                                                                                                                                              
    def plot(self):
        for i in range(len(self.movies)):
            movie = self.movies[i]  
            print("\nPlotting figures...", movie.movie_name)            
                  
            movie.plot_image(movie.folder_path)
            movie.plot_histogram(movie.folder_path)
#            if self.drift_corr == 'y':
#                self.plot_drift(path)
            movie.plot_traces(movie.folder_path)    
             
def main():
    data = Data()
    data.read()
    data.analyze()
    data.plot()

if __name__ == "__main__":
    main()





