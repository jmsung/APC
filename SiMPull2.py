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
import shutil
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
import time
from scipy.optimize import minimize


noise_cutoff = 0.5

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
    return (np.exp(-(x-b)/a)/a) * (0.5*(np.sign(x-b)+1))+1e-100

def Exp2_cutoff(a, b, c, d, x):
    return (c*(np.exp(-(x-d)/a)/a) + (1-c)*(np.exp(-(x-d)/b)/b)) * (0.5*(np.sign(x-d)+1))+1e-100   

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

        
class Mol(object):
    def __init__(self, I, row, col, spot_size):  
        self.row = row
        self.col = col
        s = int(spot_size/2)
        self.I_frame = np.mean(np.mean(I[:,row-s:row+s,col-s:col+s], axis=2), axis=1)
        
    def normalize(self):
        I = self.I_frame     
        I = I - np.min(I)
        I = I/np.max(I)

        bg_u = np.mean(I[I < 0.5])
        bg_b = np.mean(I[I > 0.5])
        self.I_frame = (I - bg_u)/(bg_b - bg_u) 
               
    def find_noise(self):
        self.I_avg = running_avg(self.I_frame, 3)
        noise0 = self.I_frame[1:-1]-self.I_avg
        noise1 = reject_outliers(noise0)
        self.noise = np.std(noise1)    

    def evaluate(self, SNR_min, dwell_min, dwell_max):
        blinking = 1

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
        self.SNR = []
        self.I_fit = np.zeros(len(signal))          
        for i in range(len(t_b)): 
            self.dwell.append(t_ub[i] - t_b[i])
            I_mean = np.mean(self.I_frame[t_b[i]:t_ub[i]])
            self.SNR.append(I_mean/self.noise)            
            self.I_fit[t_b[i]+1:t_ub[i]+1] = I_mean
        return True
                                         
class Movie(object):
    def __init__(self, movie_name, data_path):
        self.movie_name = movie_name
        self.movie_path = data_path + '\\' + movie_name   
        movie = Image.open(self.movie_path)
        self.n_frame = movie.n_frames
        print('Number of frames = %d' %(self.n_frame))
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
    def find_peaks(self, spot_size): 
        I = self.I_max
        self.peaks = peak_local_max(I, min_distance=int(spot_size*1.5))
        self.n_peaks = len(self.peaks[:, 1])
        print('\nFound', self.n_peaks, 'peaks. ')
        
    # Find real molecules from the peaks
    def find_mols(self, spot_size, SNR_min, dwell_min, dwell_max): 
        row = self.peaks[::-1,0]
        col = self.peaks[::-1,1]
        self.mols = []
        self.dwells = []
        self.noise = []
        self.SNR = []
        for i in range(self.n_peaks):
            mol = Mol(self.I, row[i], col[i], spot_size)
            mol.normalize()
            mol.find_noise()
            if mol.evaluate(SNR_min, dwell_min, dwell_max) is True:
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
                                                                                                                                                                                            
class Data(object):
    def __init__(self):
        self.data_path = os.getcwd()
        path_split = self.data_path.split('\\')      
        self.data_name = path_split[len(path_split)-1]      
        self.file_list = os.listdir(self.data_path) 
        for i in range(len(self.file_list)):
            if self.file_list[i][-3:] == 'tif':
                self.movie_name = self.file_list[i]
                print('Movie name = ', self.movie_name)
        self.movie = Movie(self.movie_name, self.data_path)
        self.tpf = float(input('Time per frame [s]? '))
        self.n_corr = int(input('How many frames for correlation [100]? '))
        self.spot_size = int(input('Spot size in pixel [3]? '))
        #self.SNR_min = int(input('Signal to noise cutoff [10]? '))
        self.SNR_min = 10
        self.dwell_min = int(input('Minimum dwell cutoff in frame [3]? ')) 
        self.dwell_max = int(input('Maximum dwell cutoff in frame [100]? ')) 
        self.auto_bin = input('Automatic bin size [y/n]? ')
        if self.auto_bin == 'n':
            self.bin_size = float(input('Bin size? ')) 
                                                                                                                                                                                              
    def analysis(self):
        movie = self.movie
        movie.I_min = np.min(movie.I, axis=0)
        movie.offset() # Pixel-wise bg subtraction, from the minimum intensity of movie 
        movie.I_max = np.max(movie.I, axis=0)
        movie.find_peaks(self.spot_size)
        movie.find_mols(self.spot_size, self.SNR_min, self.dwell_min, self.dwell_max)
        movie.find_dwelltime()
        movie.dwell_mean = np.mean(movie.dwells)-min(movie.dwells)
        movie.dwell_std = np.std(movie.dwells)
        movie.correlation(self.spot_size, self.n_corr)
                                                                                                                                                        
    def plot_fig1(self):    
        # Figure 1 - Image                                                    
        fig1 = plt.figure(1, figsize = (20, 10), dpi=300)    
        
        sp1 = fig1.add_subplot(131)  
        sp1.imshow(self.movie.I_min, cmap=cm.gray)
        sp1.set_title('Offset')
        
        sp2 = fig1.add_subplot(132)  
        sp2.imshow(self.movie.I_max, cmap=cm.gray)
        sp2.set_title('Projected max intensity') 
        
        sp3 = fig1.add_subplot(133) 
        sp3.imshow(self.movie.I_max, cmap=cm.gray)
        for j in range(len(self.movie.mols)):
            sp3.plot(self.movie.mols[j].col, self.movie.mols[j].row, 'ro', markersize=2, alpha=0.5)  
        title = 'Molecules = %d, Spot size = %d' % (len(self.movie.mols), self.spot_size)  
        sp3.set_title(title)      
        
        fig1.savefig('Fig1_Image.png')   


    def plot_fig2(self):    
        # Figure 2 - Histogram: Single Exp           
        fig2 = plt.figure(2, figsize = (20, 10), dpi=300)  
        
        sp1 = fig2.add_subplot(121)  
        if self.auto_bin == 'n':
            bins = np.arange(self.dwell_min, max(self.movie.dwells), self.bin_size)   
        else:            
            bins = 'scott'          
                                   
        hist_lifetime = sp1.hist(self.movie.dwells, bins, normed=False, color='k', histtype='step', linewidth=2)
  
        n_lifetime = len(self.movie.dwells)*(hist_lifetime[1][1] - hist_lifetime[1][0])
        x_lifetime = np.linspace(0, max(self.movie.dwells), 1000)
        y_mean = n_lifetime*Exp_cutoff(self.movie.dwell_mean, self.dwell_min, x_lifetime) 
        y_fit = n_lifetime*Exp_cutoff(self.movie.dwell_fit1[0], self.dwell_min, x_lifetime)
        sp1.plot(x_lifetime, y_mean, 'b', x_lifetime, y_fit, 'r', linewidth=2)  
        title = '\nMean dwell time [s] = %.2f +/- %.2f (N = %d)' % (self.tpf*self.movie.dwell_mean, self.tpf*self.movie.dwell_std/(len(self.movie.dwells)**0.5), len(self.movie.dwells))
        sp1.set_title(title)
        print(title)
             
        sp2 = fig2.add_subplot(122) 
        hist_lifetime = sp2.hist(self.movie.dwells, bins, normed=False, color='k', histtype='step', linewidth=2)
        sp2.set_yscale('log')
        sp2.semilogy(x_lifetime, y_mean, 'b', x_lifetime, y_fit, 'r', linewidth=2)
        sp2.axis([0, 1.1*max(x_lifetime), 0.5, 2*max(y_mean)])
        title = 'Mean dwell time [frame] = %.1f +/- %.1f (N = %d)' % (self.movie.dwell_mean, self.movie.dwell_std/(len(self.movie.dwells)**0.5), len(self.movie.dwells))
        sp2.set_title(title)

        fig2.savefig('Fig2_HistExp1.png')

        
    def plot_fig3(self):         
        # Figure 3 - Histogram: Double Exp
        fig3 = plt.figure(3, figsize = (20, 10), dpi=300)  
        
        fig3.savefig('Fig3_HistExp2.png')


    def plot_fig4(self):  
        # Figure 4: Peak intensity
        fig4 = plt.figure(4, figsize = (20, 10), dpi=300)   
        self.row = 5
        self.col = 4  
        for i in range(self.row*self.col):        
            sp = fig4.add_subplot(self.row, self.col, i+1)  
            sp.plot(self.movie.I_peak[i], 'k-')
            sp.axhline(y=0, color='b', linestyle='dashed', linewidth=1)
            sp.axhline(y=1, color='b', linestyle='dashed', linewidth=1)
          
        fig4.savefig('Fig4_PeakIntensity.png')
                
    def plot_fig5(self):  
        # Figure 5: Individual correlation   
        fig5 = plt.figure(5, figsize = (20, 10), dpi=300)  
        for i in range(self.row*self.col):        
            sp = fig5.add_subplot(self.row, self.col, i+1)  
            sp.plot(self.movie.corr[i], 'k-')    
            sp.axhline(y=0, color='b', linestyle='dashed', linewidth=1) 

        fig5.savefig('Fig5_IndivCorr.png')  
        
    def plot_fig6(self):  
        # Figure 6: Correlation with a single exponential
        fig6 = plt.figure(6, figsize = (20, 10), dpi=300)         
        corr = np.arange(1, self.n_corr+1)

        p1, pcov1 = curve_fit(exp1, corr, self.movie.corr_mean, p0=[0, self.movie.dwell_mean, 1], sigma=self.movie.corr_sem)  
        x_fit = np.linspace(0, max(corr), 1000)
        y_fit1 = exp1(x_fit, p1[0], p1[1], p1[2])     
        self.scale1 = y_fit1[0]
        self.offset1 = p1[0]
        self.time1 = p1[1]
        y_fit1 = (y_fit1 - self.offset1)/(self.scale1 - self.offset1)
        self.movie.corr_mean1 = (self.movie.corr_mean - self.offset1)/(self.scale1 - self.offset1)    

        sp1 = fig6.add_subplot(121)
        sp1.plot(self.tpf*corr, self.movie.corr_mean1, 'ko', mfc='none')        
        sp1.plot(self.tpf*x_fit, y_fit1, 'r', linewidth=2)     
        sp1.set_xlim([0, max(self.tpf*corr)])  
        sp1.set_ylim([-0.1, 1])     
        title = "Correlation time [s] = %.2f +/- %.2f (N = %d)" %(
                self.tpf*p1[2], self.tpf*pcov1[2,2]**0.5, self.movie.n_peaks)    
        sp1.set_title(title)
        sp1.set_xlabel('Time [s]')
        sp1.set_ylabel('Correlation [AU]')
        print(title)
        
        sp2 = fig6.add_subplot(122)
        sp2.plot(corr, self.movie.corr_mean1, 'ko', mfc='none')
        sp2.set_yscale('log')
        sp2.semilogy(x_fit, y_fit1, 'r', linewidth=2)    
        sp2.set_xlim([0, max(corr)])  
        sp2.set_ylim([min(y_fit1)/10, 1])  
        title = "Correlation time [frame] = %.1f +/- %.1f, Tpf [s] = %.2f" %(
                p1[2], pcov1[2,2]**0.5, self.tpf)   
        sp2.set_title(title) 
        sp2.set_xlabel('Lag time [frame]')
        sp2.set_ylabel('Correlation [AU]')
              
        fig6.savefig('Fig6_CorrExp1.png')
                
    def plot_fig7(self):  
        # Figure 7: Correlation with a double exponential
        fig7 = plt.figure(7, figsize = (20, 10), dpi=300)        
        corr = np.arange(1, self.n_corr+1) 

        #def exp2(x, a, b, c, d, e):
            #return a + b * np.exp(-x/c) + d* np.exp(-x/e)  
        bnds = (0.001, 1000) 
        p2, pcov2 = curve_fit(exp2, corr, self.movie.corr_mean, p0=[self.offset1, self.scale1, self.time1, self.scale1/10, self.time1*2], sigma=self.movie.corr_sem, bounds=bnds)  
        x_fit = np.linspace(0, max(corr), 1000)
        y_fit2 = exp2(x_fit, p2[0], p2[1], p2[2], p2[3], p2[4])     
        scale2 = y_fit2[0]
        offset2 = p2[0]
        y_fit2 = (y_fit2 - offset2)/(scale2 - offset2)
        self.movie.corr_mean2 = (self.movie.corr_mean - offset2)/(scale2 - offset2)    

        sp1 = fig7.add_subplot(121)
        sp1.plot(self.tpf*corr, self.movie.corr_mean2, 'ko', mfc='none')        
        sp1.plot(self.tpf*x_fit, y_fit2, 'r', linewidth=2)   
        sp1.set_xlim([0, max(self.tpf*corr)])  
        sp1.set_ylim([-0.1, 1])   
        title = "Correlation time [s] = %.2f +/- %.2f (%d %%), %.2f +/- %.2f (%d %%) (N = %d)" %(
                self.tpf*p2[2], self.tpf*pcov2[2,2]**0.5, 100*p2[1]/(p2[1]+p2[3]), 
                self.tpf*p2[4], self.tpf*pcov2[4,4]**0.5, 100*p2[3]/(p2[1]+p2[3]),
                self.movie.n_peaks)    
        sp1.set_title(title)
        sp1.set_xlabel('Time [s]')
        sp1.set_ylabel('Correlation [AU]')
        print(title)
        
        sp2 = fig7.add_subplot(122)
        sp2.plot(corr, self.movie.corr_mean2, 'ko', mfc='none')
        sp2.set_yscale('log')
        sp2.semilogy(x_fit, y_fit2, 'r', linewidth=2)  
        sp2.set_xlim([0, max(corr)])  
        sp2.set_ylim([min(y_fit2)/2, 1])     
        title = "Correlation time [frame] = %.1f +/- %.1f (%d %%), %.1f +/- %.1f (%d %%), Tpf [s] = %.2f" %(
                p2[2], pcov2[2,2]**0.5, 100*p2[1]/(p2[1]+p2[3]), 
                p2[4], pcov2[4,4]**0.5, 100*p2[3]/(p2[1]+p2[3]),
                self.tpf)   
        sp2.set_title(title) 
        sp2.set_xlabel('Lag time [frame]')
        sp2.set_ylabel('Correlation [AU]')

        fig7.savefig('Fig7_CorrExp2.png')
             
    def plot_traces(self):                                                                                                                                                                                                                                                                  
        # Figure for individual traces    
        save_trace = input('Save individual traces [y/n]? ')
        if save_trace == 'y':    
            percent = int(input('How much percent [1-100]? '))    
            if percent < 1:
                percent = 1
            if percent > 100:
                percent = 100
                
            i_fig = 1               
            n_col = 3
            n_row = 3
            directory = self.data_path+'\\Figures'
            if os.path.exists(directory):
                shutil.rmtree(directory)
                os.makedirs(directory)
            else:
                os.makedirs(directory)
                
            n_fig = int(len(self.movie.mols)*percent/100)
                
            for j in range(n_fig):
                k = j%(n_col*n_row)
                if k == 0:                      
                    fig = plt.figure(100, figsize = (25, 15), dpi=300)
                sp = fig.add_subplot(n_row, n_col, k+1)
                sp.plot(self.movie.frame, self.movie.mols[j].I_frame, 'k.', linewidth=0.5, markersize=3)
                sp.plot(self.movie.frame, self.movie.mols[j].I_s, 'b', linewidth=1, markersize=3)
                sp.plot(self.movie.frame, self.movie.mols[j].I_fit, 'r', linewidth=2, markersize=3)
                sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)
                sp.axhline(y=noise_cutoff, color='k', linestyle='dotted', linewidth=1)
                sp.axhline(y=1, color='k', linestyle='dotted', linewidth=1)                
                title_sp = '(%d, %d) (noise = %.2f)' % (self.movie.mols[j].row, self.movie.mols[j].col, self.movie.mols[j].noise)
                sp.set_title(title_sp)
                fig.subplots_adjust(wspace=0.3, hspace=0.5)
                if (k == n_col*n_row-1) | (j == n_fig-1):
                    print("Save Fig %d (%d %%)" % (i_fig, (j/len(self.movie.mols))*100+1))
                    fig.savefig(directory+"\\Fig%d.png" % (i_fig)) 
                    fig.clf()
                    i_fig += 1

    def plot(self):
        print("\nPlotting figures...")
        self.plot_fig1()
        self.plot_fig2()
        self.plot_fig3()
        self.plot_fig4()
        self.plot_fig5()
        self.plot_fig6()
        self.plot_fig7()   
        self.plot_traces()    
             
def main():
    data = Data()
    data.analysis()
    data.plot()

if __name__ == "__main__":
    main()

# Crashing issue

# Automatic cutoff [mean/4, mean*10]
# MLE 2 exp

# MLE error
# There are two errors to fix

# Drift correction from spatial correlation

# Automatic correlation range

# What if there are two reaction paths. Check in theory and simulation



