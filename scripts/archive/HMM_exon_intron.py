"""
# HMM test: Exon, Intron

What is a hidden Markov model? by Sean R Eddy

Start > E > 5 > I > End

Transition probability
tp (Start > E) = 1.0
tp (E > E) = 0.9
tp (E > 5) = 0.1
tp (5 > I) = 1.0
tp (I > I) = 0.9
tp (I > End) = 0.1

Emission probability
E: A (0.25), C (0.25), G (0.25), T (0.25)
5: A (0.05), C (0.00), G (0.95), T (0.00)
I: A (0.40), C (0.10), G (0.10), T (0.40)  

"""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt

# Transition probability [S E 5 I E]
tr_prob = [[0.0, 1.0, 0.0, 0.0, 0.0],
           [0.0, 0.9, 0.1, 0.0, 0.0],
           [0.0, 0.0, 0.0, 1.0, 0.0],
           [0.0, 0.0, 0.0, 0.9, 0.1],
           [0.0, 0.0, 0.0, 0.0, 1.0]]

# Emission probability for E/5/I in A/T/G/C
em_prob = [[0.25, 0.25, 0.25, 0.25],
           [0.05, 0.00, 0.95, 0.00],
           [0.45, 0.05, 0.10, 0.40]]
     
state_path = 'EEEEEEEEEEEEEEEEEE5IIIIIII'
sequence = 'CTTCATGTGAAAGCAGACGTAAGTCA'  

p_E = 0.99
p_I = 0.99

class HMM:
    def __init__(self):
        pass
        
    def read(self):
        self.state_path = state_path 
        self.sequence = sequence     
        self.tr_prob = tr_prob
        self.em_prob = em_prob  
        self.five = state_path.find('5')

    def generate(self):
        self.state_path = 'E'
        while np.random.rand() < p_E:
            self.state_path += 'E'
        self.state_path += '5I'
        while np.random.rand() < p_I:
            self.state_path += 'I'
        print(self.state_path)

        self.sequence = ''
        for i in range(len(self.state_path)):
            if self.state_path[i] == 'E':
                self.sequence += np.random.choice(['A', 'T', 'G', 'C'], 1, p = em_prob[0])[0]
            elif self.state_path[i] == '5':
                self.sequence += np.random.choice(['A', 'T', 'G', 'C'], 1, p = em_prob[1])[0]
                self.five = i
            elif self.state_path[i] == 'I':
                self.sequence += np.random.choice(['A', 'T', 'G', 'C'], 1, p = em_prob[2])[0]        
            else:
                pass      
        print(self.sequence)  


    def find(self, x):
        if x == 'A':
            return 0
        elif x == 'T':
            return 1
        elif x == 'G':
            return 2
        elif x == 'C':
            return 3
        else:
            return 4            
        
        
    def analyze(self):
        num_nt = len(self.sequence)
        self.log_P = np.zeros(num_nt-2)
        for i in range(1, num_nt-1):
            log_tp = 0
            for j in range(num_nt-1):
                if j < i-1:
                    log_tp += np.log(p_E)
                elif j == i-1:
                    log_tp += np.log(1-p_E)
                elif j == i:
                    log_tp += np.log(1.0)
                else:
                    log_tp += np.log(p_I)
            log_tp += np.log(1-p_I)
                    

            
#            log_tp = (np.log(tr_prob[1][1])*(i-1) 
#                    + np.log(tr_prob[1][2])
#                    + np.log(tr_prob[2][3])
#                    + np.log(tr_prob[3][3])*(num_nt-i-3)
#                    + np.log(tr_prob[3][4]))

            log_ep = 0            
            for j in range(num_nt):
                if j < i:
                    log_ep += np.log(em_prob[0][self.find(self.sequence[j])])
                elif j == i:
                    log_ep += np.log(em_prob[1][self.find(self.sequence[j])])
                else:
                    log_ep += np.log(em_prob[2][self.find(self.sequence[j])])
                    
                    
            self.log_P[i-1] = log_tp + log_ep  
#            self.log_P[i-1] = log_ep                

        self.P = np.exp(self.log_P)
        self.P = self.P/sum(self.P)        
        self.result = self.five == np.argmax(self.P)+1
                                 
    def plot(self):
        plt.close('all')
        fig = plt.figure()    
                
        sp = fig.add_subplot(121)  
        sp.plot(range(1, len(self.P)+1), self.P, 'k.')
        sp.axvline(x=self.five, color='r', linestyle='dashed', lw=1)
        sp.set_title('%s' %(['Fail', 'Success'][int(self.result)]))                
                
        sp = fig.add_subplot(122)  
        sp.semilogy(range(1, len(self.P)+1), self.P, 'k.')
        sp.axvline(x=self.five, color='r', linestyle='dashed', lw=1)
        sp.set_title('%s' %(['Fail', 'Success'][int(self.result)]))  

        plt.show()
        

            


def main():
    hmm = HMM()
#    hmm.read()
    hmm.generate()
    hmm.analyze()
    hmm.plot()

if __name__ == "__main__":
    main()

















