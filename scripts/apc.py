"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created by Jongmin Sung (jongmin.sung@gmail.com)

Single molecule binding and unbinding analysis for anaphase promoting complex (apc) 

class Data() 
- path, name, load(), list[], n_movie, movies = [Movie()], plot(), analysis(), spot_size, frame_rate, 
- path, name, n_row, n_col, n_frame, pixel_size, 
- background, spots = [], molecules = [Molecule()]

class Molecule()
- position (row, col), intensity 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path  
import os
import shutil
#from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import rank

#import platform
from hmmlearn import hmm
import sys
sys.path.append("../apc/apc")   # Path where apc_config and apc_funcs are located
from apc_config import data_dir # Configuration 


# User-defined functions
from apc_funcs import read_movie, running_avg, reject_outliers, str2bool, \
Exp_cutoff, find_dwelltime, flatfield_correct, drift_correct, offset_correct


# User input ----------------------------------------------------------------

#directory = data_dir/'18-12-04 D-box mutant'
directory = data_dir/'18-06-29 Drift correction'
directory = data_dir

#frame_rate = 33
#n_trace = 20

noise_cutoff = 0.5
#spot_size = 3
SNR_min = 10

# ---------------------------------------------------------------------------

class Mol:
    def __init__(self, I, row, col, spot_size):  
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

        if self.noise > 1/SNR_min: 
            return False        
        
        x = running_avg(self.I_frame, 3)
        self.I_s = np.array([x[0]]+x.tolist()+[x[-1]])       
        signal = self.I_s > noise_cutoff

        t_b = []
        t_ub = []
        for i in range(len(signal)-1):
            if (signal[i]==False) & (signal[i+1]==True):
                t_b.append(i)
            if (signal[i]==True) & (signal[i+1]==False):
                t_ub.append(i)
        
        if len(t_b)*len(t_ub) == 0: 
            return False 
        if t_ub[0] < t_b[0]: # remove pre-existing binding
            del t_ub[0]
        if len(t_b)*len(t_ub) == 0: 
            return False                
        if t_ub[-1] < t_b[-1]: # remove unfinished binding
            del t_b[-1]
        if len(t_b)*len(t_ub) == 0: 
            return False      

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

        if len(t_b)*len(t_ub) == 0: 
            return False    
              
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
                    
        # HMM      
        X = self.I_frame.reshape(len(signal), 1)
            
        # Set a new model for traidning
        remodel = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)        
        
        # Set initial parameters for training
        remodel.startprob_ = np.array([5.0, 0.5])
        remodel.transmat_ = np.array([[0.9, 0.1], 
                                      [0.1, 0.9]])
        remodel.means_ = np.array([0, 1])
        remodel.covars_ = np.tile(np.identity(1), (2, 1, 1)) * self.noise**2        
           
        # Estimate model parameters (training)
        remodel.fit(X)  
        
        # Find most likely state sequence corresponding to X
        Z_predict = remodel.predict(X)
        
        # Reorder state number such that X[Z=0] < X[Z=1] 
        if X[Z_predict==0].mean() > X[Z_predict==1].mean():
            Z_predict = 1 - Z_predict
            remodel.transmat_ = np.array(
                [[remodel.transmat_[1][1], remodel.transmat_[1][0]],
                 [remodel.transmat_[0][1], remodel.transmat_[0][0]]])
   
        self.tp_ub = remodel.transmat_[0][1] 
        self.tp_bu = remodel.transmat_[1][0]
        
        # Sequence of predicted states        
        self.I_predict = np.zeros(len(X))
        for i in range(2):
            self.I_predict[Z_predict==i] = X[Z_predict==i].mean()  
                          
        return True
                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
class Movie:
    def __init__(self, path):
        self.path = path
        self.dir = path.parent
        self.name = path.name

    # Read info.txt and movie.tif   
    def read(self):  
        # Read info.txt
        self.info = {}
        with open(Path(self.dir/'info.txt')) as f:
            for line in f:
                line = line.replace(" ", "") # remove white space
                if line == '\n': # skip empty line
                    continue
                (key, value) = line.rstrip().split("=")
                self.info[key] = value
        self.dt = float(self.info['time_interval'])
        self.spot_size = int(float(self.info['magnification'])*3/100) 

        # Read original image (I_original) from movie.tif
        self.bin_size = 20
        self.I_original, self.metadata = read_movie(self.path, self.bin_size)
        self.I_min = np.min(self.I_original, axis=0)
        self.n_frame = np.size(self.I_original, axis=0)
        self.n_row = np.size(self.I_original, axis=1)
        self.n_col = np.size(self.I_original, axis=2) 

    def correct(self):
        # Drift correct
        if str2bool(self.info['drift_correct']) == True:
            print('drift_correct = True')
            self.drift_row, self.drift_col, self.I_drift \
                = drift_correct(self.I_original)
        else:
            print('drift_correct = False')
            self.drift_row = np.zeros((self.n_frame))
            self.drift_col = np.zeros((self.n_frame))
            self.I_drift = np.array(self.I_original)
        
        # Flatfield correct
        if str2bool(self.info['flatfield_correct']) == True:
            print('flatfield_correct = True')
            self.I_bin, self.I_fit, self.I_flatfield \
                = flatfield_correct(self.I_drift, self.bin_size)
        else:
            print('flatfield_correct = False')
            self.I_bin = np.zeros((self.n_row, self.n_col))
            self.I_fit = np.zeros((self.n_row, self.n_col))
            self.I_flatfield = np.array(self.I_drift)

        self.I = np.array(self.I_flatfield)

    def analyze(self):
        # Find peaks from local maximum
        self.peaks = peak_local_max(self.I_max0, min_distance=int(self.spot_size*1.5))        
        self.n_peaks = len(self.peaks[:, 1])
        self.row = self.peaks[::-1,0]
        self.col = self.peaks[::-1,1]
        print('Found', self.n_peaks, 'peaks. ')    
        
        # Find molecules    
        self.mols = []
        self.dwells = []
        self.noise = []
        self.SNR = []
        self.tp_ub = []
        self.tp_bu = []
        
        for i in range(self.n_peaks):
            mol = Mol(self.I, self.row[i], self.col[i], self.spot_size)
            mol.normalize()
            mol.find_noise()
#           if mol.evaluate(SNR_min, dwell_min, dwell_max) is True:      
            if mol.evaluate() is True:
                self.mols.append(mol)    
                self.dwells.extend(mol.dwell)
                self.noise.append(mol.noise)
                self.SNR.extend(mol.SNR)
                self.tp_ub.append(mol.tp_ub)
                self.tp_bu.append(mol.tp_bu)                
                       
        print('Found', len(self.mols), 'molecules. \n')          
#        find_mols()#spot_size, SNR_min)#, self.dwell_min, self.dwell_max)
        self.dwell_fit1 = find_dwelltime(self.dwells)
        self.dwell_mean = np.mean(self.dwells)-min(self.dwells)
        self.dwell_std = np.std(self.dwells)
  
    def plot_image1_max(self):
        fig = plt.figure(figsize = (20, 10), dpi=300)            

        sp = fig.add_subplot(131)  
        I_original = np.max(self.I_original, axis=0)
        sp.imshow(I_original, cmap=cm.inferno)
        sp.set_title('Max intensity - original')

        sp = fig.add_subplot(132)  
        I_drift = np.max(self.I_drift, axis=0)
        sp.imshow(I_drift, cmap=cm.inferno)
        sp.set_title('Max intensity - drift')

        sp = fig.add_subplot(133)  
        I_flatfield = np.max(self.I_flatfield, axis=0)
        sp.imshow(I_flatfield, cmap=cm.inferno)
        sp.set_title('Max intensity - flatfield')        

        fig.savefig(self.dir/'image1_max.png')   
        plt.close(fig)                                                                                                                                                                                                                                                                                                                                                                                                                                                            

    def plot_image2_drift(self):                                                     
        fig = plt.figure(figsize = (20, 10), dpi=300)    

        sp = fig.add_subplot(211)  
        sp.plot(self.drift_row)
        sp.set_title('Drift in row')

        sp = fig.add_subplot(212)  
        sp.plot(self.drift_col)
        sp.set_title('Drift in col')

        fig.savefig(self.dir/'image2_drift.png')   
        plt.close(fig)

    def plot_image3_flatfield(self):                                                     
        fig = plt.figure(figsize = (20, 10), dpi=300)    

        sp = fig.add_subplot(131)  
        sp.imshow(self.I_bin, cmap=cm.inferno)
        sp.set_title('Max intensity - bin')
 
        sp = fig.add_subplot(132)  
        sp.imshow(self.I_fit, cmap=cm.inferno)
        sp.set_title('Max intensity - fit')

        sp = fig.add_subplot(133)  
        sp.imshow(self.I_bin - self.I_fit, cmap=cm.inferno)
        sp.set_title('Max intensity - dev')

        fig.savefig(self.dir/'image3_flatfield.png')   
        plt.close(fig)




#        sp3 = fig1.add_subplot(133) 
#        sp3.imshow(self.I_max, cmap=cm.gray)
#        for j in range(len(self.mols)):
#            sp3.plot(self.mols[j].col, self.mols[j].row, 'ro', ms=2, alpha=0.5)  
#        title = 'Molecules = %d, Spot size = %d' % (len(self.mols), self.spot_size)  
#        sp3.set_title(title)      
        

    def plot_histogram(self):            
        fig2 = plt.figure(2, figsize = (20, 10), dpi=300)  
            
        sp1 = fig2.add_subplot(121)                                 
        hist_lifetime = sp1.hist(self.dwells, bins='scott', density=False, 
                                 color='k', histtype='step', linewidth=2)    
        n_lifetime = len(self.dwells)*(hist_lifetime[1][1] - hist_lifetime[1][0])
        x_lifetime = np.linspace(0, max(self.dwells), 1000)
        y_mean = n_lifetime*Exp_cutoff(self.dwell_mean, 1, x_lifetime) 
        y_fit = n_lifetime*Exp_cutoff(self.dwell_fit1[0], 1, x_lifetime)
        sp1.plot(x_lifetime, y_mean, 'b', x_lifetime, y_fit, 'r', linewidth=2)  
        title = '\nMean dwell time [s] = %.2f +/- %.2f (N = %d)' \
              % (self.dt*self.dwell_mean, \
                self.dt*self.dwell_std/(len(self.dwells)**0.5), len(self.dwells))
        sp1.set_title(title)
        print(title)
                
        sp2 = fig2.add_subplot(122) 
        hist_lifetime = sp2.hist(self.dwells, bins='scott', density=False, 
                                 color='k', histtype='step', linewidth=2)
        sp2.set_yscale('log')
        sp2.semilogy(x_lifetime, y_mean, 'b', x_lifetime, y_fit, 'r', linewidth=2)
        sp2.axis([0, 1.1*max(x_lifetime), 0.5, 2*max(y_mean)])
        title = 'Mean dwell time [frame] = %.1f +/- %.1f (N = %d)' \
              % (self.dwell_mean, self.dwell_std/(len(self.dwells)**0.5), \
                len(self.dwells))
        sp2.set_title(title)
        fig2.savefig(self.dir/'Histogram.png')
        plt.close(fig2)
                    
    def plot_traces(self):                                                                                                                                                                                                                                                                                              
        trace_dir = self.dir/'Traces'
        if os.path.exists(trace_dir):
            shutil.rmtree(trace_dir)
            os.makedirs(trace_dir)
        else:
            os.makedirs(trace_dir)
                
        frame = np.arange(self.n_frame)
                
        n_fig = min(self.num_trace, len(self.mols))        
        for j in range(n_fig):                 
            fig = plt.figure(100, figsize = (25, 15), dpi=300)
            sp = fig.add_subplot(111)
            sp.plot(frame, self.mols[j].I_frame, 'k', lw=0.5, ms=3)
    #        sp.plot(frame, self.mols[j].I_s, 'b', linewidth=1, markersize=3)
            sp.plot(frame, self.mols[j].I_fit, 'b', lw=1, ms=3)
            sp.plot(frame, self.mols[j].I_predict, 'r', lw=1, ms=3)            
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)
            sp.axhline(y=noise_cutoff, color='k', linestyle='dotted', lw=1)
            sp.axhline(y=1, color='k', linestyle='dotted', lw=1)                
            title_sp = '(%d, %d) (noise = %.2f)' \
            % (self.mols[j].row, self.mols[j].col, self.mols[j].noise)
            sp.set_title(title_sp)
            fig.subplots_adjust(wspace=0.3, hspace=0.5)
            print("Save Trace %d (%d %%)" % (j+1, ((j+1)/n_fig)*100))
            fig_name = 'Trace%d.png' %(j+1)
            fig.savefig(trace_dir/fig_name) 
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
        im2 = sp2.imshow(movie.drift[frame], cmap=cm.gray, extent = [-r,r,r,-r])
        plt.colorbar(im2)
        sp2.set_title('Correlation (Frame = %d)' %(frame))
        
        sp3 = fig8.add_subplot(223)
        sp3.plot(movie.drift_x, 'k')
        sp3.set_title('Drift in X')  
    
        sp4 = fig8.add_subplot(224)
        sp4.plot(movie.drift_y, 'k')
        sp4.set_title('Drift in Y') 
    
        fig8.savefig(self.movie_dir/'Fig8_Drift.png')   
        plt.close(fig8)
                  
    def plot_HMM(self):

        log_tp_ub = np.log(self.tp_ub)
        log_tp_bu = np.log(self.tp_bu)
        
#        log_tp_ub = reject_outliers(np.log(self.tp_ub))
#        log_tp_bu = reject_outliers(np.log(self.tp_bu))        
        
        time_u = 1/np.exp(np.mean(log_tp_ub))
        time_b = 1/np.exp(np.mean(log_tp_bu))

        fig3 = plt.figure(3, figsize = (20, 10), dpi=300)   
        
        sp = fig3.add_subplot(121)  
        sp.hist(log_tp_bu)
        sp.set_xlabel("Log(TP_bu)")
        sp.set_title("Bound time = %.1f" %(np.mean(time_b)))        
        
        sp = fig3.add_subplot(122)  
        sp.hist(log_tp_ub)
        sp.set_xlabel("Log(TP_ub)")
        sp.set_title("Unbound time = %.1f" %(np.mean(time_u))) 
        
        fig3.savefig(self.movie_dir/'HMM.png')  
        plt.close(fig3)     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    def save(self):
        pass        
      

                       
def main():
    # Find all the movies (*.tif) in the directory tree
    movie_paths = list(directory.glob('**/*.tif'))
    print('%d movies are found' %(len(movie_paths)))

    # Run through each movie
    for i, movie_path in enumerate(movie_paths):
        print('='*100)
        print('Movie #%d/%d' %(i+1, len(movie_paths)))
        print('Path:', movie_path.parent)
        print('Name:', movie_path.name)

        # Check info.txt exist.
        info_file = Path(movie_path.parent/'info.txt')
        if not info_file.exists():
            print('info.txt is not found.')
            continue

        # Make a movie instance
        movie = Movie(movie_path)

        # Read the movie
        movie.read()

        # Flatfield and drift correction
        movie.correct()

        # Analyze the movie
#        movie.analyze()

        # Save the result into result.txt
        movie.save()

        # Plot the result
        print("\nPlotting figures...")  
        movie.plot_image1_max()
        movie.plot_image2_drift()          
        movie.plot_image3_flatfield()
              
#        movie.plot_histogram()
#        movie.plot_traces()
#        movie.plot_HMM()


if __name__ == "__main__":
    main()



"""
To-do

* set dwell_max, dwell_min
* Histogram of intensity
* HMM step finding
* Classes of binding & unbinding events

* Save results in a text
* Seperate code to read text and combine or compare

-----------------------------------------------------------------
Done

* read tiff meta data (19-05-31)
* find movies in the subfolder (19-06-06)
* read info.txt for the parameters (19-06-06)
* Flat field correction: max > bin > fit > norm (19-06-12)
* Drift correction (19-06-15)
"""



