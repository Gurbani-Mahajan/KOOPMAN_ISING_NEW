# using koopman operator to detetct phase transition
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy.special import eval_hermite
from numba import njit

@njit(cache=True)

def energy_2d(s, J):
    l = len(s)
    E_ij = 0
    for i in range(l):
        for j in range(l):
            E_ij -= J * s[i, j] * (s[(i + 1) % l, j] + s[i, (j + 1) % l] + s[(i - 1) % l, j] + s[i, (j - 1) % l])
    return E_ij / 2

def random_lattice(num_sites):
    s = np.ones([num_sites, num_sites])
    # randomising spins so that net magnetic moment is roughly 0
    for i in range(num_sites):
        for j in range(num_sites):
            if random.random() > 0.5:
                s[i, j] = 1
            else:
                s[i, j] = -1
    return s

def correlations(s):
    n=len(s)
    sum_1=0
    sum_2=0
    for i in range(n):
        for j in range(n):
            #C1
            sum_1+= s[i, j] * (s[(i + 1) % n, j] + s[i, (j + 1) % n])
            #C2
            sum_2+= s[i, j] * (s[(i + 2) % n, j] + s[i, (j + 2) % n])
    # average of all configs
    C1=sum_1/(2*n*n)
    C2=sum_2/(2*n*n)
    return C1, C2

def glauber(s, T):
    s_2d = s.copy()
    E_initial = energy_2d(s_2d,J)
    E_glauber= []
    C1=[]
    C2=[]
    M = []  # magnetisation
    n = len(s_2d)
    sweeps=int(1e4)
    sweep_size=n**2
    steps = sweep_size*sweeps
    #running parallel chains to check GR statistic
    N=4 #no. of chains
    means=[]
    c0=[]
    W=0
    burn_in=1000
    '''
    # generating time series
    for i in range(sweeps):
        for j in range (sweep_size):
            r = random.randrange(n)  # random row
            c = random.randrange(n)  # random column
            s_new = s_2d[r, c] * (-1)  # flipping the spin and storing new spin
            neighbours = s_2d[r, (c - 1) % n] + s_2d[r, (c + 1) % n] + s_2d[(r - 1) % n, c] + s_2d[(r + 1) % n, c]  # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
            dE = 2 * J * s_2d[r, c] * neighbours  # change in energy on spin flip
            E_new = E_initial + int(dE)  # new value of energy
            # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
            if np.random.rand() < 1 / (1 + np.exp(dE / T)):
                E_initial = E_new  # accepting flip
                s_2d[r, c] = s_new
        # correlations for edmd observables (just E and M are non-markovian as multiple configs can have same M and E)
        c1,c2 = correlations(s_2d)
        C1.append(c1)
        C2.append(c2)
    C1=np.array(C1)
    C2=np.array(C2)
    '''
    for i in range(N):
        c0=[]
        for j in range(sweeps):
            for k in range(sweep_size):
                r = random.randrange(n)  # random row
                c = random.randrange(n)  # random column
                s_new = s_2d[r, c] * (-1)  # flipping the spin and storing new spin
                neighbours = s_2d[r, (c - 1) % n] + s_2d[r, (c + 1) % n] + s_2d[(r - 1) % n, c] + s_2d[(r + 1) % n, c]  # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
                dE = 2 * J * s_2d[r, c] * neighbours  # change in energy on spin flip
                E_new = E_initial + int(dE)  # new value of energy
                # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
                if np.random.rand() < 1 / (1 + np.exp(dE / T)):
                    E_initial = E_new  # accepting flip
                    s_2d[r, c] = s_new
            
            c1,c2=correlations(s_2d)
            if i==0:
                C1.append(c1)
                C2.append(c2)
            if j>=burn_in:
                c0.append(c1)
            #print('done')
            if j%int(1e3)==0:
                print('step = ',j,', chain = ',i)
        c0=np.array(c0)
        #for between chain variance
        c0_mean=np.mean(c0)
        means.append(c0_mean)
        #for within chain variance
        s0=np.sum((c0-c0_mean)**2)/(sweeps-1)
        W+=s0
    B=np.sum((means-np.mean(means))**2)*(sweeps/(N-1)) #between chain variance adding original chain
    V=((sweeps-1)/sweeps)*W + B/sweeps
    R_hat=np.sqrt(V/W)
    C1=np.array(C1)
    C2=np.array(C2)
    return C1,C2,R_hat


def herm_observables (X,Y,degree):
    #initialising dictionary
    #psi=np.zeros((X.shape[0],(degree+1)**2))
    psi=[]

    #standardising
    X_mean=np.mean(X)
    Y_mean=np.mean(Y)
    X_std=np.std(X)
    Y_std=np.std(Y)
    if X_std == 0:
        X_std=1
    if Y_std == 0:
        Y_std=1
    x = ((X- X_mean)/X_std)/np.sqrt(2)
    y = ((Y- Y_mean)/Y_std)/np.sqrt(2)
    
    #building dictionary such that the rows=ith snapshot of C1 and C2, cols= prod of hermite polynomials ( of C1 and C2) of upto jth degree distinct pairs
    for i in range(degree+1):
        herm_x=eval_hermite(i,x)
        for j in range (degree+1):
            herm_y=eval_hermite(j,y)
            psi.append(herm_x*herm_y)
    psi=np.column_stack(psi)
    print(psi.shape)
    return psi


def edmd(psi_X,psi_Y,m):
    print(psi_X.shape)
    G = (psi_X.T @ psi_X) / m  # edmd gram matrix 
    A = (psi_X.T @ psi_Y) / m  # edmd cross-covariance matrix
    K = np.linalg.lstsq(G + 1e-6 * np.eye(psi_X.shape[1]), A, rcond=None)[0]  # matrix was ill-conditioned and division by zero errors arose. had to use matrix norm
    eigvals = np.linalg.eigvals(K)
    eigvals = np.sort(np.abs(eigvals))[::-1]  # sorted eigenvalues (descending)
    return eigvals



num_sites = 16  # number of lattice sites
n=num_sites
num_spins= num_sites*num_sites
n_2=num_spins
J = 1  # ferromagnetic
kb = 1  # boltzman constant in units

# discrete time evolution of state of the model using Koopman operator (edmd)

# defining parameter (temperature range)
T = np.linspace(2,3,50)
T_relax=[]
herm_gaps_1=[]
herm_gaps_2=[]
corr_gaps_1=[]
corr_gaps_2=[]
relax = []
gr1=np.zeros(len(T))
gr2=np.zeros(len(T))
for k in range(len(T)):
    t = T[k]
    s_2d_1=random_lattice(num_sites)
    #s_2d_2=random_lattice(num_sites*2)
    C11,C12,gr1[k]= glauber(s_2d_1, t)
    #C21,C22,gr2[k] = glauber(s_2d_2, t)
    # pairs of successive magnetizations, energies and correlations
    X11=C11[:-1]
    Y11=C11[1:]
    X12=C12[:-1]
    Y12=C12[1:]
    #X21=C21[:-1]
    #Y21=C21[1:]
    #X22=C22[:-1]
    #Y22=C22[1:]
    psi_X1 = herm_observables(X11,X12,3)
    psi_Y1 = herm_observables(Y11,Y12,3)
    #psi_X2 = corr_observables(X21,X22)
    #psi_Y2 = corr_observables(Y21,Y22)
    eigvals_c_1 = edmd(psi_X1,psi_Y1,X11.shape[0])
    #eigvals_c_2 = edmd(psi_X2,psi_Y2,X21.shape[0])
    corr_spec_gap_1 = 1 - np.abs(eigvals_c_1[1])/np.abs(eigvals_c_1[0])
    #corr_spec_gap_2 = 1 - np.abs(eigvals_c_2[1])/np.abs(eigvals_c_2[0])
    corr_gaps_1.append(corr_spec_gap_1)
    #corr_gaps_2.append(corr_spec_gap_2)
    
    if corr_spec_gap_1!=0:
        relax.append(1/corr_spec_gap_1) #relaxation time
        T_relax.append(t)
    #print('done ',k)
    

#plotting spectral gap
corr_gaps_smooth_1=savgol_filter(corr_gaps_1, 11, 3)
#corr_gaps_smooth_2=savgol_filter(corr_gaps_2, 11, 3)


plt.figure()
plt.plot(T, corr_gaps_1, label='(raw)')
plt.plot(T , corr_gaps_smooth_1, 'r--', label='(smoothed)')
plt.axvline(x=2 / np.log(1 + np.sqrt(2)),linestyle='--',label='Tc (Onsager)')
plt.xlabel('Temperature')
plt.ylabel('Spectral Gap ')
plt.grid(True)
plt.title('Koopman Spectral Gap vs Temperature (16x16 lattice)')
plt.legend()
plt.show()

'''
plt.figure()
plt.plot(T, corr_gaps_2, label='(raw)')
plt.plot(T , corr_gaps_smooth_2, 'r--', label='(smoothed)')
plt.axvline(x=2 / np.log(1 + np.sqrt(2)),linestyle='--',label='Tc (Onsager)')
plt.xlabel('Temperature')
plt.ylabel('Spectral Gap ')
plt.grid(True)
plt.title('Koopman Spectral Gap vs Temperature (32x32 lattice)')
plt.legend()
plt.show()
'''

#finding Tc
print('-------------------------------\n 16 X 16 LATTICE \n-------------------------------')
min_idx1=np.argmin(corr_gaps_1)
tc_koop1=T[min_idx1]
err_koop1= np.std(corr_gaps_1)
print('Using the least value of the Koopman spectral gap on 16 x 16 lattice, Tc = ',tc_koop1,' ± ',err_koop1)
print('Onsager Tc = 2.269')
print('Percentage error in estimate = ', (np.abs(tc_koop1 - 2.269)/2.269)*100,'%')

#convergence checks
print('GR statistic for 50 temps in order from 2->3K:')
print(gr1)

'''
print('-------------------------------\n 32 X 32 LATTICE \n-------------------------------')
min_idx2=np.argmin(corr_gaps_2)
tc_koop2=T[min_idx2]
err_koop2= np.std(corr_gaps_2)
print('Using the least value of the Koopman spectral gap on 32 x 32 lattice, Tc = ',tc_koop2,' ± ',err_koop2)
print('Onsager Tc = 2.269')
print('Percentage error in estimate = ', (np.abs(tc_koop2 - 2.269)/2.269)*100,'%')
'''

#plotting relaxation time
plt.figure()
plt.plot(T_relax, relax)
plt.axvline(x=2 / np.log(1 + np.sqrt(2)), linestyle='--', label=r'Tc (Onsager)')
plt.xlabel('Temperature')
plt.ylabel('Relaxation Time ')
plt.grid(True)
plt.title('Koopman Relaxation time vs Temperature')
plt.legend()
plt.show()

