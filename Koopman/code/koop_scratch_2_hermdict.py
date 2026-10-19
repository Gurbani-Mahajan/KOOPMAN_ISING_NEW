# using koopman operator to detetct phase transition
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy.special import eval_hermite
from tabulate import tabulate


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
            '''
            #C2
            sum_2+= s[i, j] * (s[(i + 2) % n, j] + s[i, (j + 2) % n])
            '''
            
    # average of all configs
    C1=sum_1/(2*n*n)
    #C2=sum_2/(2*n*n)
    return C1 , #C2

def glauber(s, T):
    s_2d = s.copy()
    E_initial = energy_2d(s_2d,J)
    M = []  # magnetisation
    E=[]  #energy
    #correlations
    C1=[]
    C2=[]
    n = len(s_2d)
    steps = int(1e6)
    '''
    #since for T<Tc, the chain takes longer to equilibriate
    if T<=2.269:
        steps=int(1e6)

    print(steps)
    '''
    
    mc_step = n*n #1 MC sweep = all sites visited once
    mc_cnt=0
    # generating time series
    for i in range(steps):
        r = random.randrange(n)  # random row
        c = random.randrange(n)  # random column
        s_new = s_2d[r, c] * (-1)  # flipping the spin and storing new spin
        neighbours = s_2d[r, (c - 1) % n] + s_2d[r, (c + 1) % n] + s_2d[(r - 1) % n, c] + s_2d[
            (r + 1) % n, c]  # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
        dE = 2 * J * s_2d[r, c] * neighbours  # change in energy on spin flip
        E_new = E_initial + int(dE)  # new value of energy
        # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
        if np.random.rand() < 1 / (1 + np.exp(dE / T)):
            E_initial = E_new  # accepting flip
            s_2d[r, c] = s_new
        if steps == int(1e6):
            if i%(int(1e5))==0:
                print('glauber i=',i)
        else:
            if i%(int(1e4))==0:
                print('glauber i=',i)
        c1=correlations(s_2d)
        C1.append(c1)
        #C2.append(c2)
        M.append(np.sum(s_2d))
        #E.append(E_initial)
    C1=np.array(C1)
    M=np.array(M)
    
    #convergence diagnostics

    '''
    #integrated autocorrelation time to check markovinity
    max_lag= steps
    M_mean=np.mean(M)
    sum_i=0
    for i in range(0,max_lag,mc_step):
        sum_j=0
        cnt=0
        for j in range(0,len(M),mc_step):
            if (i+j)<len(M):
                sum_j +=  (np.abs(M[j]-M_mean)*np.abs(M[j+i]-M_mean))
                cnt+=1
        
        if (i//mc_step)%1000==0:
            print('done step = ',i,', mc_step = ',i//(mc_step))
        
        acf=(sum_j)/(cnt*np.var(M))
        sum_i+=acf
    act=1+2*mc_step*sum_i
    '''
    act=0
    
    #GR statistic to check convergence to stationary distribution for C1 (4 chains, 1 original, 3 for verification)
    N=4
    means_c1=[]
    means_m=[]
    W_c1=0
    W_m=0
    burn_in=10000
    for l in range(N):
        c0=[]
        m=[]
        s_2d=random_lattice(n)
        E_initial=energy_2d(s_2d,J)
        for j in range(steps):
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
            if j>=burn_in:
                c1=correlations(s_2d)
                c0.append(c1)
                m.append(np.sum(s_2d))
        print('done')
        c0=np.array(c0)
        m=np.array(m)
        #for between chain variance
        c1_mean=np.mean(c0)
        m_mean=np.mean(m)
        means_c1.append(c1_mean)
        means_m.append(m_mean)
        #for within chain variance
        s0_c1=np.sum((c0-c1_mean)**2)/(steps-1)
        W_c1+=s0_c1
        s0_m=np.sum((m-m_mean)**2)/(steps-1)
        W_m+=s0_m
    #addding original chain (upto 1e6 iterations)
    C1_gr=C1[:int(1e6)]
    mean_C1=np.mean(C1_gr)
    s0_og_c1=np.sum((C1_gr - mean_C1)**2)/(steps-1)
    W_c1=(W_c1+s0_og_c1)/N #within chain variance adding original chain
    M_gr=M[:int(1e6)]
    mean_M=np.mean(M_gr)
    s0_og_m=np.sum((M_gr - mean_M)**2)/(steps-1)
    W_m=(W_m+s0_og_m)/N #within chain variance adding original chain
    means_c1.append(mean_C1)
    B_c1=np.sum((means_c1-np.mean(means_c1))**2)*(steps/(N-1)) #between chain variance adding original chain
    V_c1=((steps-1)/steps)*W_c1 + B_c1/steps
    R_hat_c1=np.sqrt(V_c1/W_c1)
    means_m.append(mean_M)
    B_m=np.sum((means_m-np.mean(means_m))**2)*(steps/(N-1)) #between chain variance adding original chain
    V_m=((steps-1)/steps)*W_m + B_m/steps
    R_hat_m=np.sqrt(V_m/W_m)
    
    return M,C1,R_hat_m,R_hat_c1 #,act

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
    psi=np.array(psi)
    print(psi.shape)
    return psi.T

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
T = np.linspace(2,2.538,50) #mid=2.269, steo size=0.01076
T_relax=[]
herm_gaps_1=[]
herm_gaps_2=[]
act1=np.zeros(len(T))
act2=np.zeros(len(T))
gr_c1=np.zeros(len(T))
gr_m=np.zeros(len(T))
relax = []
for k in range(len(T)):
    t = T[k]
    s_2d_1=random_lattice(num_sites)
    #s_2d_2=random_lattice(num_sites*2)
    M,C1,gr_m[k],gr_c1[k] = glauber(s_2d_1, t)
    
    #,act2[k],gr2[k] = glauber(s_2d_2, t)
    # pairs of successive magnetizations, energies and correlations
    x1 = M[:-1]
    y1 = M[1:]
    x2 = C1[:-1]
    y2 = C1[1:]
    psi_x1 = herm_observables(x1,x2,3)
    psi_y1= herm_observables(y1,y2,3)
    #psi_x2 = herm_observables(x2,5)
    #psi_y2= herm_observables(y2,5)
    eigvals_h_1 = edmd(psi_x1, psi_y1, x1.shape[0])
    #eigvals_h_2 = edmd(psi_x2, psi_y2, x2.shape[0])
    if np.any((eigvals_h_1-1)>(1e-4)):
        print('Nuissance eigenvalue found at T = ',t, ' for 16x16 lattice. λ = ',eigvals_h_1[np.where(eigvals_h_1>1)])
    #if np.any(eigvals_h_2>1):
        #print('Nuissance eigenvalue found at T = ',t, ' for 32x32 lattice. λ = ',eigvals_h_2[np.where(eigvals_h_2>1)])
    herm_spec_gap_1 = 1 - np.abs(eigvals_h_1[1])
    #herm_spec_gap_2 = 1 - np.abs(eigvals_h_2[1])
    herm_gaps_1.append(herm_spec_gap_1)
    #herm_gaps_2.append(herm_spec_gap_2)
    
    if herm_spec_gap_1!=0:
        relax.append(1/herm_spec_gap_1) #relaxation time
        T_relax.append(t)
    print('done ',k)
    
#finding Tc
herm_gaps_smooth_1= savgol_filter(herm_gaps_1, 11, 3)
#herm_gaps_smooth_2=savgol_filter(herm_gaps_2, 11, 3)

print('-------------------------------\n 16 X 16 LATTICE \n-------------------------------')
min_idx=np.argmin(herm_gaps_1)
min_idx_smooth=np.argmin(herm_gaps_smooth_1)
tc_koop=T[min_idx]
tc_koop_smooth=T[min_idx_smooth]
err_koop= np.std(herm_gaps_1)
print('Using the least value of the Koopman spectral gap, Tc = ',tc_koop,' ,or using smoothed curve, Tc= ',tc_koop_smooth)
print('Onsager Tc = 2.269')
print('Percentage error in estimate = ', (np.abs(tc_koop - 2.269)/2.269)*100,'%')

#convergence checks
print('GR statistic for M of 50 temps in order from 2->2.538K:')
print(gr_m)

#convergence checks
print('GR statistic for C1 of 50 temps in order from 2->2.538K:')
print(gr_c1)

#plotting spectral gap

plt.figure()
plt.plot(T, herm_gaps_1, label='hermite dict (raw)')
plt.plot(T , herm_gaps_smooth_1, 'r--', label='hermite dict (smoothed)')
plt.axvline(x=2 / np.log(1 + np.sqrt(2)),linestyle='--',label='Tc (Onsager)')
plt.xlabel('Temperature')
plt.ylabel('Spectral Gap ')
plt.grid(True)
plt.title('Koopman Spectral Gap vs Temperature (16x16)')
plt.legend()
plt.show()

'''
print('-------------------------------\n 32 X 32 LATTICE \n-------------------------------')
min_idx0=np.argmin(herm_gaps_2)
min_idx_smooth0=np.argmin(herm_gaps_smooth_2)
tc_koop0=T[min_idx0]
tc_koop_smooth0=T[min_idx_smooth0]
err_koop0= np.std(herm_gaps_2)
print('Using the least value of the Koopman spectral gap, Tc = ',tc_koop0,' ,or using smoothed curve, Tc= ',tc_koop_smooth)
print('Onsager Tc = 2.269')
print('Percentage error in estimate = ', (np.abs(tc_koop0 - 2.269)/2.269)*100,'%')

plt.figure()
plt.plot(T, herm_gaps_2, 'g', label='hermite dict (raw)')
plt.plot(T , herm_gaps_smooth_2, 'm--', label='hermite dict (smoothed)')
plt.axvline(x=2 / np.log(1 + np.sqrt(2)),linestyle='--',label='Tc (Onsager)')
plt.xlabel('Temperature')
plt.ylabel('Spectral Gap ')
plt.grid(True)
plt.title('Koopman Spectral Gap vs Temperature (32x32)')
plt.legend()
plt.show()

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


