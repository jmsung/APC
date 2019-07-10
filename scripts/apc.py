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
import seaborn as sns
from pathlib import Path  
import os
import shutil
#from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from skimage.filters import median
from skimage.morphology import disk
from skimage.filters import rank
from scipy.stats import norm
#import sys
#sys.path.append("../apc/apc")   # Path where apc_config and apc_funcs are located
#from apc_config import data_dir # Configuration 
from skimage.filters.rank import entropy
from skimage.morphology import disk
from tifffile import TiffFile
from inspect import currentframe, getframeinfo
from imreg_dft.imreg import translation

from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import disk
from skimage.filters import rank
from skimage.filters.rank import entropy
from skimage.filters import threshold_sauvola, threshold_niblack, threshold_local
from scipy.ndimage import gaussian_filter

# User-defined functions
#from apc_funcs import read_movie, running_avg, reject_outliers, str2bool, \
#flatfield_correct, drift_correct, reject_outliers, get_trace, fit_trace

fname = getframeinfo(currentframe()).filename # current file name
current_dir = Path(fname).resolve().parent
data_dir = Path(fname).resolve().parent.parent/'data' 

# User input ----------------------------------------------------------------

directory = data_dir
#directory = data_dir/'19-05-29 Movies 300pix300pi'

# ---------------------------------------------------------------------------

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def check_outliers(I):
    dev = np.abs(I - np.median(I))
    mdev = np.median(dev)
    s = dev/mdev if mdev else 0.
    return s < 3


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

        # Read movie.tif
        with TiffFile(self.path) as tif:
            imagej_hyperstack = tif.asarray()
            imagej_metadata = str(tif.imagej_metadata)
            self.metadata = imagej_metadata.split(',')

        # write meta_data    
        with open(self.path.parent/'meta_data.txt', 'w') as f:
            for item in imagej_metadata:
                f.write(item+'\n')

        # Crop the image to make the size integer multiple of 10
        self.bin_size = 20
        self.n_frame = np.size(imagej_hyperstack, 0)
        n_row = np.size(imagej_hyperstack, 1)
        self.n_row = int(int(n_row/self.bin_size)*self.bin_size)        
        n_col = np.size(imagej_hyperstack, 2) 
        self.n_col = int(int(n_col/self.bin_size)*self.bin_size)
        self.I_original = imagej_hyperstack[:,:self.n_row,:self.n_col]

        print('[frame, row, col] = [%d, %d, %d]' %(self.n_frame, self.n_row, self.n_col))  


    def drift_correct(self):
        # Drift correct
        if str2bool(self.info['drift_correct']) == True:
            print('drift_correct = True')

            I = self.I_original.copy()
            I_ref = I[int(len(I)/2),] # Mid frame as a reference frame

            # Translation as compared with I_ref
            d_row = np.zeros(len(I), dtype='int')
            d_col = np.zeros(len(I), dtype='int')
            for i, I_frame in enumerate(I):
                result = translation(I_ref, I_frame)
                d_row[i] = round(result['tvec'][0])
                d_col[i] = round(result['tvec'][1])      

            # Changes of translation between the consecutive frames
            dd_row = d_row[1:] - d_row[:-1]
            dd_col = d_col[1:] - d_col[:-1]

            # Sudden jump in translation set to zero
            step_limit = 1
            dd_row[abs(dd_row)>step_limit] = 0
            dd_col[abs(dd_col)>step_limit] = 0

            # Adjusted translation
            d_row[0] = 0
            d_col[0] = 0
            d_row[1:] = np.cumsum(dd_row)
            d_col[1:] = np.cumsum(dd_col)

            # Offset mid to zero
            self.drift_row = d_row - int(np.median(d_row))
            self.drift_col = d_col - int(np.median(d_col))

            # Translate images
            self.I_drift = self.I_original.copy()
            for i in range(len(I)):
                self.I_drift[i,] = np.roll(self.I_original[i,], self.drift_row[i], axis=0)
                self.I_drift[i,] = np.roll(self.I_original[i,], self.drift_col[i], axis=1)        
        else:
            print('drift_correct = False')
            self.drift_row = np.zeros((self.n_frame))
            self.drift_col = np.zeros((self.n_frame))
            self.I_drift = np.array(self.I_original)
        

    def flatfield_correct(self):

        self.I_drift_max = np.max(self.I_drift, axis=0)

        # Flatfield correct
        if str2bool(self.info['flatfield_correct']) == True:
            print('flatfield_correct = True')

            # Masking from local threshold        
            self.mask = self.I_drift_max > threshold_local(self.I_drift_max, block_size=31, offset=-31) 
            self.I_mask = self.I_drift_max*self.mask

            # Local averaging signals
            self.I_bin = np.zeros((self.n_row, self.n_col))
            m = self.bin_size
            for i in range(int(self.n_row/m)):
                for j in range(int(self.n_col/m)):
                    window = self.I_mask[i*m:(i+1)*m, j*m:(j+1)*m].flatten()          
                    signals = [signal for signal in window if signal > 0]
                    if signals:
                        self.I_bin[i*m:(i+1)*m,j*m:(j+1)*m] = np.mean(signals)

            # Fill empty pixels with the mean of its neighbors 
            for i in range(int(self.n_row/m)):
                for j in range(int(self.n_col/m)):
                    if self.I_bin[i*m,j*m] == 0:
                        window = self.I_bin[max(0,(i-1)*m):min((i+2)*m,self.n_row), max(0,(j-1)*m):min((j+2)*m,self.n_col)].flatten() 
                        signals = [signal for signal in window if signal > 0]
                        if signals:
                            self.I_bin[i*m:(i+1)*m,j*m:(j+1)*m] = np.mean(signals)   

            self.I_bin[self.I_bin==0] = np.mean(self.I_bin[self.I_bin>0])


            # Smoothening 
            self.I_bin_filter = gaussian_filter(self.I_bin, sigma=10)

            # Flatfield correct
            self.I_flatfield = np.array(self.I_drift)
            for i in range(self.n_frame):
                self.I_flatfield[i,] = self.I_drift[i,] / self.I_bin_filter * np.max(self.I_bin_filter)    

        else:
            print('flatfield_correct = False')
            self.mask = np.zeros((self.n_row, self.n_col))
            self.I_mask = np.zeros((self.n_row, self.n_col))
            self.I_bin = np.zeros((self.n_row, self.n_col))
            self.I_bin_filter = np.zeros((self.n_row, self.n_col))
            self.I_flatfield = self.I_drift.copy()

        self.I = self.I_flatfield.copy()
        self.I_max = np.max(self.I, axis=0)


    def find_spotss(self):
        # Find local maxima from I_max
        self.peaks = peak_local_max(self.I_max, min_distance=int(self.spot_size*1.0))        
        self.n_peaks = len(self.peaks[:, 1])
        self.row = self.peaks[::-1,0]
        self.col = self.peaks[::-1,1]

        # Get the intensity of the peaks
        s = int((self.spot_size-1)/2) # Half-width of spot size
        self.I_peaks = np.zeros((self.n_peaks))
        for i in range(self.n_peaks):
            self.I_peaks[i] = np.mean([
                self.I_max[j, k] # Mean intensity in ROI
                    for j in range(self.row[i]-s,self.row[i]+s+1) 
                    for k in range(self.col[i]-s,self.col[i]+s+1) 
            ])
      
        # Find outliers based on the intensity 
        self.good_spots = check_outliers(self.I_peaks) 
        print('Found', len(self.good_spots), 'peaks. ')     
        print('Rejected', sum(~self.good_spots), 'outliers.')           

    def find_mols(self):

        # Fit intensity traces
        self.I_trace = np.zeros((self.n_peaks, self.n_frame), dtype='float')
        self.I_fit = np.zeros((self.n_peaks, self.n_frame), dtype='float')
        self.I_rmsd = np.zeros((self.n_peaks), dtype='float')
        self.tp_ub = np.zeros((self.n_peaks), dtype='float')
        self.tp_bu = np.zeros((self.n_peaks), dtype='float')
        self.dwell = [[], [], [], []]
        self.wait = [[], [], [], []]
  
#        for i in range(self.n_peaks):
#            self.I_trace[i] = get_trace(self.I, self.row[i], self.col[i], self.spot_size)
#            self.I_fit[i], self.tp_ub[i], self.tp_bu[i] = fit_trace(self.I_trace[i])
#            self.I_rmsd[i] = (np.mean((self.I_trace[i]-self.I_fit[i])**2.0))**0.5
#
#        self.I_trace_all = np.ravel(self.I_trace) 


    def plot_clean(self):
        # clean existing png files in the folder
        files = os.listdir(self.dir)    
        for file in files:
            if file.endswith('png'):
                os.remove(self.dir/file)    


    def plot_image0_min_max_original(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(20, 10), ncols=2, nrows=2)

        I_min = np.min(self.I_original, axis=0)
        I_max = np.max(self.I_original, axis=0)

        sp1 = ax1.imshow(I_min, cmap='gray')
        fig.colorbar(sp1, ax=ax1) 
        ax2.hist(I_min.ravel(), 20, histtype='step', lw=1, color='k')    
        ax2.set_yscale('log')
        ax2.set_xlim(0, np.max(I_max)) 
        ax2.set_title('Min intensity - original')

        sp3 = ax3.imshow(I_max, cmap='gray')
        fig.colorbar(sp3, ax=ax3) 
        ax4.hist(I_max.ravel(), 50, histtype='step', lw=1, color='k')                      
        ax4.set_yscale('log')
        ax4.set_xlim(0, np.max(I_max)) 
        ax4.set_title('Max intensity - original')

        fig.tight_layout()
        fig.savefig(self.dir/'image0_min_max.png')   
        plt.close(fig)                                                                                                                                                                                                                                                                                                                                                                                                                                                            


    def plot_image1_cross_section(self):
        I_row = np.squeeze(self.I_original[:,int(self.n_row/2),:])
        I_col = np.squeeze(self.I_original[:,:,int(self.n_row/2)])

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=(20, 10), ncols=2, nrows=2)

        sp1 = ax1.imshow(I_row, cmap='gray')
        ax1.set_xlabel('Row')
        ax1.set_ylabel('Frame')

        sp2 = ax2.imshow(I_col, cmap='gray')
        ax2.set_xlabel('Col')
        ax2.set_ylabel('Frame')

        ax3.plot(np.max(I_row, axis=0), 'ko-')
        ax3.set_xlim([0, self.n_row])
        ax3.set_xlabel('Row')

        ax4.plot(np.max(I_col, axis=0), 'ko-')
        ax4.set_xlim([0, self.n_col])
        ax4.set_xlabel('Col')

        fig.tight_layout()
        fig.savefig(self.dir/'image1_cross_section.png')   
        plt.close(fig)         


    def plot_image2_drift(self):                                                     
        fig = plt.figure(figsize = (20, 10), dpi=300)    

        sp = fig.add_subplot(211)  
        sp.plot(self.drift_row, 'k')
        sp.set_title('Drift in row')

        sp = fig.add_subplot(212)  
        sp.plot(self.drift_col, 'k')
        sp.set_title('Drift in col')

        fig.tight_layout()
        fig.savefig(self.dir/'image2_drift.png')   
        plt.close(fig)


    def plot_image3_flatfield(self):                                                     
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(figsize=(20, 10), ncols=3, nrows=2)

        sp1 = ax1.imshow(self.I_drift_max, cmap=cm.gray)
        fig.colorbar(sp1, ax=ax1) 
        ax1.set_title('Max intensity - original')      
  
        sp2 = ax2.imshow(self.mask, cmap=cm.gray)
        ax2.set_title('Mask')           

        sp3 = ax3.imshow(self.I_mask, cmap=cm.gray)
        fig.colorbar(sp3, ax=ax3) 
        ax3.set_title('Max intensity - Mask')

        sp4 = ax4.imshow(self.I_bin, cmap=cm.gray)
        fig.colorbar(sp4, ax=ax4) 
        ax4.set_title('Intensity - bin')

        sp5 = ax5.imshow(self.I_bin_filter, cmap=cm.gray)
        fig.colorbar(sp5, ax=ax5) 
        ax5.set_title('Intensity - bin filter')        

        sp6 = ax6.imshow(self.I_max, cmap=cm.gray)
        fig.colorbar(sp6, ax=ax6) 
        ax6.set_title('Max intensity - flatfield')

        fig.tight_layout()
        fig.savefig(self.dir/'image3_flatfield.png')   
        plt.close(fig)


    def plot_image4_peaks_max(self):
        fig = plt.figure(figsize = (20, 10), dpi=300)     

        sp = fig.add_subplot(1,2,1)         
        sp.imshow(self.I_max, cmap=cm.gray)
        color = [['b','r'][int(i)] for i in self.good_spots] 
        sp.scatter(self.col, self.row, lw=0.8, s=50, facecolors='none', edgecolors=color)
        sp.set_title('Max intensity at peaks')  

        sp = fig.add_subplot(1,2,2)    
        bins = np.linspace(min(self.I_peaks), max(self.I_peaks), 30)     
        sp.hist(self.I_peaks, bins = bins, histtype='step', lw=2, color='b')
        sp.hist(self.I_peaks[self.good_spots], bins = bins, histtype='step', lw=2, color='r')
        sp.set_title('Max intensity distribution')  

        fig.tight_layout()
        fig.savefig(self.dir/'image4_peaks_max.png')   
        plt.close(fig)


    def plot_image5_trace_all(self):
        fig = plt.figure(figsize = (20, 10), dpi=300)     
#            x = np.linspace(np.min(signal), np.max(signal), 100)
#            param = norm.fit(signal)     
#            pdf = norm.pdf(x, loc = param[0], scale = param[1])          
        sp = fig.add_subplot(1,2,1)  
        sp.hist(self.I_trace_all, 50, histtype='step', lw=2)
        sp = fig.add_subplot(1,2,2)  
        sp.hist(self.I_trace_all, 50, histtype='step', lw=2)
        sp.set_yscale('log')
#            sp.plot(x, pdf, 'r', lw=2)
        sp.set_title('All intensities at peaks')  
        fig.savefig(self.dir/'image5_trace_all.png')   
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
        # Make a Trace folder                                                                                                                                                                                                                                                                                         
        trace_dir = self.dir/'Traces'
        if os.path.exists(trace_dir): # Delete if already existing 
            shutil.rmtree(trace_dir)
        os.makedirs(trace_dir)
                
        frame = np.arange(self.n_frame)
                
        n_fig = min(num_trace, len(self.I_trace))        
        for i in range(n_fig):                 
            fig = plt.figure(100, figsize = (25, 15), dpi=300)
            sp = fig.add_subplot(111)
            sp.plot(frame, self.I_trace[i], 'k', lw=2)
            sp.plot(frame, self.I_fit[i], 'b', lw=2)                      
#            title_sp = '(%d, %d) (noise = %.2f)' \
#            % (self.mols[j].row, self.mols[j].col, self.mols[j].noise)
#            sp.set_title(title_sp)
            fig.subplots_adjust(wspace=0.3, hspace=0.5)
            print("Save Trace %d (%d %%)" % (i+1, ((i+1)/n_fig)*100))
            fig_name = 'Trace%d.png' %(i+1)
            fig.savefig(trace_dir/fig_name) 
            fig.clf()
           

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
    movie_paths = [fn for fn in directory.glob('**/*.tif')
                   if not fn.name == 'GFP.tif']

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
            print('info.txt does not exist.')
            continue

        # Make a movie instance
        movie = Movie(movie_path)

        # Read the movie
        movie.read()

        # Flatfield and drift correction
        movie.drift_correct()
        movie.flatfield_correct()

        # Find spots
        movie.find_spots()

        # Find molecules
        movie.find_mols()

        # Save the result into result.txt
        movie.save()

        # Plot the result
        print("\nPlotting figures...")  
        movie.plot_clean()
        movie.plot_image0_min_max_original()        
        movie.plot_image1_cross_section()
        movie.plot_image2_drift()          
        movie.plot_image3_flatfield()        
        movie.plot_image4_peaks_max()
#        movie.plot_image5_trace_all()              
#        movie.plot_histogram()
#        movie.plot_traces()
#        movie.plot_HMM()


if __name__ == "__main__":
    main()



"""
To-do

* Better step finding algorithm > hmm vs smoothening+thresholding
* Krammer barrier crossing 
* Classes of binding & unbinding events
* set dwell_max, dwell_min
* Save results in a text
* Seperate code to read text and combine or compare


