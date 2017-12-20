"""
Analysis of SiMPull data (Jongmin Sung)

class Data() 
- path, data_name, load(), movie_list[], movie_num, movies = [Movie()], plot()

class Movie() 
- movie_name, frame_number, pixels = [Pixel()], molecules = [Molecule()]

class Pixel()
- pixel_position, pixel_intensity

class Molecule()
- mol_number, mol_position 





- cell_name, intensity[array], analysis(), bg(), rescale(), cm(), median_center()
- radial_intensity()


To-do
- Class structure
- Open image and load data
- mol find algorithm? per pixel or per frame?
- find mol? SNR per pixel from all frames > mol per pixel > cluster neighbor pixels
- find mol? molecule per frame > combine from all frames



"""

from __future__ import division, print_function, absolute_import
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import scipy

color = 'magma'
scale_factor = 100
r_step = 6

class Cell(object):
    def __init__(self, cell_name, sample_path):
        self.cell_name = cell_name
        self.cell_path = sample_path + '\\' + cell_name   
        self.intensity = np.array(Image.open(self.cell_path))
        self.intensity = self.intensity - self.intensity.min() # BG subtraction
        
    def mc(self): # Median center
        I = self.intensity.copy()
        I = I/scale_factor
        Ix = []; Iy = []
        for i in range(np.size(I, axis=0)):
            for j in range(np.size(I, axis=1)):
                Ii = [i]*int(I[i,j]); Ix += Ii
                Ij = [j]*int(I[i,j]); Iy += Ij
        Ix_mc = int(np.median(Ix))
        Iy_mc = int(np.median(Iy))
        self.mc= [Ix_mc, Iy_mc]
        I[Ix_mc-1:Ix_mc+1, :] = I.max()
        I[:, Iy_mc-1:Iy_mc+1] = I.max()
        self.intensity_mc = I.copy()       
                       
                        
    def radial_intensity(self):
        I = self.intensity.copy()
        x_max = np.size(I, axis=0)
        y_max = np.size(I, axis=1) 
        x_mc = self.mc[0]
        y_mc = self.mc[1]
        r_max = np.max([x_mc, x_max-x_mc, y_mc, y_max-y_mc])
        self.r = np.arange(r_step, r_max, r_step)
        self.Ir = np.zeros(len(self.r))
        self.Is = np.zeros(len(self.r))
        for i in range(len(self.r)):
            ix = []; iy = []
            for x in range(x_max):
                for y in range(y_max):
                    dist = ((x-x_mc)**2.0 + (y-y_mc)**2.0)**0.5                 
                    if dist < self.r[i]:
                        if i == 0:
                            ix.append(x); iy.append(y)
                        else:
                            if dist > self.r[i-1]:
                                ix.append(x); iy.append(y) 
            I_select = I[ix, iy]
            self.Ir[i] = I_select.mean()
            self.Is[i] = I_select.std()
        
        p = [max(self.Ir), 50, 20, 0]   
#        self.popt, self.pcov = curve_fit(error_func, self.r, self.Ir, sigma=1/self.Is**2.0)
        self.popt, self.pcov = curve_fit(error_func, self.r, self.Ir, p0=p)
        print(self.popt)

     
class Sample(object):
    def __init__(self, sample_name, data_path):
        self.sample_name = sample_name
        self.sample_path = data_path + '\\' + sample_name
        self.cell_list = os.listdir(self.sample_path)
        self.cell_num = len(self.cell_list)
        self.cells = []   
        for cell_name in self.cell_list:
            cell = Cell(cell_name, self.sample_path)
            self.cells.append(cell)

class Data(object):
    def __init__(self):
        self.data_path = os.getcwd()
        path_split = self.data_path.split('\\')
        self.data_name = path_split[len(path_split)-1]
        self.sample_list = os.listdir(self.data_path) 
        self.sample_num = len(self.sample_list)
        self.samples = []   
        for sample_name in self.sample_list:
            sample = Sample(sample_name, self.data_path)
            self.samples.append(sample)
                       
    def analysis(self):
        for i in range(self.sample_num):
            sample = self.samples[i]
            cell_num = sample.cell_num
            for j in range(cell_num):
                cell = sample.cells[j]
                cell.mc()    
                cell.radial_intensity()
                print(j)
                       
                
    def plot(self):
        plt.close('all')
        # Plot overall cell images
        for i in range(self.sample_num):
            plt.figure(self.sample_list[i])
            sample = self.samples[i]
            cell_num = sample.cell_num
            for j in range(cell_num):
                row = np.floor((cell_num/1.5)**0.5)
                col = np.ceil(cell_num/row)
                plt.subplot(row, col, j+1)
                cell = sample.cells[j]
                plt.imshow(cell.intensity_mc, color)
                plt.title(cell.cell_name[-8:-4])
            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
        # Plot analysis of individual cells
        for i in range(self.sample_num):
            sample = self.samples[i]
            cell_num = sample.cell_num
            for j in range(cell_num):
                cell = sample.cells[j]
                fig = plt.figure(cell.cell_name)
                
                sp1 = fig.add_subplot(221)
                sp1.imshow(cell.intensity, color); 
                #sp1.colorbar()  
                
                sp2 = fig.add_subplot(222)
                sp2.imshow(cell.intensity_mc, color) 
                
                sp3 = fig.add_subplot(223)
                sp3.errorbar(cell.r, cell.Ir, yerr=cell.Is, fmt='ko', ecolor='k')
                x = np.linspace(0, max(cell.r)*1.1, 100)
                y = error_func(x, cell.popt[0], cell.popt[1], cell.popt[2], cell.popt[3])
                sp3.plot(x, y, 'r')
                sp3.axis([0, max(cell.r), 0, max(cell.Ir+cell.Is)])         
                
                sp4 = fig.add_subplot(224)
                sp4.hist([cell.intensity], bins='scott', normed=False, 
                        color='k', histtype='step', linewidth=2); 
                sp4.set_yscale("log")     
            plt.subplots_adjust(wspace=0.3, hspace=0.5)                                         
        plt.show()
                      
                          
# Start  
data = Data()
data.analysis()
data.plot()