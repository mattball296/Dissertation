# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:33:02 2023

@author: mattb
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import levy_stable as stable
import sympy as sp

j = complex(0,1)


class OneSidedTemperedStable:
    '''
    Class to represent one sided tempered stable distributions as defined in definition 3.2.1
    The shift gamma is 0, and alpha=1 is not supported
    '''
    
    
    def __init__(self, alpha, beta, lamda):
        ''' initialise the distribution with parameters alpha, beta and lambda'''
        try:
            if alpha==1:
                raise ValueError
            self.alpha = alpha #tail index
            self.beta = beta #scale from the Levy measure
            self.lamda = lamda #controls tempering
        except ValueError:
            print("alpha=1 is not supported")
      
  
    def _cf(self, y):
        ''' function which calculates the characteristic function for the distribution using the representation derived in lemma 3.2.6'''
        a = self.alpha 
        b = self.beta
        l = self.lamda 
        exp = gamma(-a)*b*((l - j*y)**a -l**a + j*y*a*l**(a-1))
        return np.exp(exp)
    
# for symbolic differentiation

    def sympycf(self, y):
        ''' symbolic representation of the characteristic function for differentiation'''
        y = sp.symbols('y')
        a = self.alpha
        b = self.beta
        l = self.lamda
        exp = gamma(-a) * b * ((l - sp.I * y) ** a - l ** a + sp.I * y * a * l ** (a - 1))
        return sp.exp(exp)
    
    def differentiate_sympycf(self, n):
        '''calculates nth derivative of the characteristic function'''
        y = sp.symbols('y')
        cf = self.sympycf(y)
        derivative = cf
        for _ in range(n):
            derivative = derivative.diff(y)
        return derivative

    def nth_moment(self, n):
        '''uses the nth derivative to calculate the nth moment of the distribution'''
        derivative = self.differentiate_sympycf(n)
        derivative_at_zero = derivative.subs('y', 0)
        return derivative_at_zero / (sp.I ** n)


    '''2 possible ways to evaluate the pdf'''

    def _pdf1(self, x):
        ''' evaluates the pdf using numerical Fourier inversion (not recommended)'''
        integral = lambda y: np.exp(-j*y*x)*self._int_cf(y)
        ans = (1/(2*np.pi) * quad(integral, -np.inf, np.inf)[0]).real
        return ans
    
    def _pdf2(self, x, t=1):
        '''evaluates the pdf using relation (16) (recommended)'''
        
        def F(z, a , b , l):
            return b*np.exp(-l*z)/(z**(a+1))
        

        a = self.alpha
        b = self.beta
        l = self.lamda
        
        d = quad(lambda z : z*F(z,a,b,0),1,np.inf)[0]
        
        c = (-b*gamma(-a)*np.cos(np.pi*a*0.5))**(1/a)
        
        dist = stable(a,1,d,c)

        D = quad(lambda z : z*(1-np.exp(-l*z))*F(z,a,b,0), 0 ,1)[0]
        shift = quad(lambda z : z*F(z,a,b,l),1,np.inf)[0] - D
        
        expct = np.exp(-(c*l)**a/np.cos(0.5*a*np.pi) - d*l)
        
        return np.exp(-l*(x + shift))*dist.pdf(x + shift)/expct 
     
    def ck(self):
        
        '''function to calculate c and k from algorithm 2'''
        b = self.beta
        l = self.lamda
        a = self.alpha
    
        phi = lambda y: abs(self._cf(y))
        phi2 = lambda y: abs(-self._cf(y)*b*(b*gamma(1-a)**2*(l**(a-1)-(l-y*j)**(a-1))**2 + gamma(2-a)*(l-j*y)**(a-2)))
        
        I1, _ = quad(phi, -np.inf, np.inf)
        I2, _ = quad(phi2, -np.inf, np.inf)
    
        c = 1/(2*np.pi)*I1
        k = 1/(2*np.pi)*I2
        return c, k
  
    def q(self, z, t=1):
        c, k = self.ck(t)
        return min(c, k/(z**2))
  
    def _exact_sim(self, c=None, k=None):
        ''' function which implements algorithm 2 to exactly sample from the distribution'''
        counter = 0
        
        if c==None and k==None:
            ck = self.ck()
            c = ck[0]
            k = ck[1]
            
    
        while True:
    
            counter += 1
    
            U1 = np.random.uniform(-1, 1)
            U2 = np.random.uniform(-1, 1)
    
            V = np.sqrt(float(k)/float(c))*(float(U1)/float(U2))
            U = np.random.uniform(0,1)
    
            if abs(V) < np.sqrt(float(k)/float(c)):
                if c*U < self._pdf2(V):
                    return V, counter
            else:
                if k*U < V**2*self._pdf2(V):
                    return V, counter
                
   
    def rvs(self, nsamples):
        ''' generates 'nsamples' samples and returns the values and average execution time'''
        samples = np.zeros(nsamples)
        counts = np.zeros(nsamples)
        c, k = self.ck()
        for i in range(nsamples):
            if i%10000 == 0:
                print(i)
            y, count = self.exact_sim(c, k)
            samples[i] = y
            counts[i] = count
        return samples, np.mean(counts)