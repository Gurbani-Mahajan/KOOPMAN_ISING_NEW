# using wang-landau mc algo to solve 2d ising model
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter


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


def glauber(J,s, T):
    s_2d = s.copy()
    E_initial = energy_2d(s_2d,J)
    M = []  # magnetisation
    E=[]  #energy
    n = len(s_2d)
    sweeps = int(1e4)

    #since for T<Tc, the chain takes longer to equilibriate
    #if T<=2.269:
        #sweeps=int(1e)
    
    sweep_size = n*n #1 MC sweep = all sites visited once
    mc_cnt=0
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
    
    #running 4 parallel chains to get GR statistic to check convergence
    N=4
    means=[]
    m0=[]
    W=0
    burn_in=2000
    for i in range(N):
        m=[]
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
            if i==0:
                M.append(np.sum(s_2d))
                E.append(energy_2d(s_2d,J))
            if j>=burn_in:
                m.append(np.sum(s_2d))
            if j%(sweeps/10)==0:
                print('step = ',j,', chain = ',i)
        #for between chain variance
        m_mean=np.mean(m)
        means.append(m_mean)
        #for within chain variance
        s0=np.sum((m-m_mean)**2)/(sweeps-1)
        W+=s0
    #addding original chain (upto 1e4 iterations)
    B=np.sum((means-np.mean(means))**2)*(sweeps/(N-1)) #between chain variance adding original chain
    V=((sweeps-1)/sweeps)*W + B/sweeps
    R_hat=np.sqrt(V/W)
    M=np.array(M)
    E=np.array(E)
    return M,E,R_hat #,act

'''
def thermalise(s,T,E,ln_g):
    s_eq=s.copy()
    n = len(s)
    E_initial = int(energy_2d(s_eq, J))
    step = 5000
    for i in range(step):
        r = random.randrange(n)  # random row
        c = random.randrange(n)  # random column
        s_new = s_eq[r, c] * (-1)  # flipping the spin and storing new spin
        neighbours = s_eq[r, (c - 1) % n] + s_eq[r, (c + 1) % n] + s_eq[(r - 1) % n, c] + s_eq[(r + 1) % n, c]
        dE = 2 * J * s_eq[r, c] * neighbours  # modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
        E_new = E_initial + int(dE)  # new value of energy
        ind = E_new - E_min
        # checking to see if energy value is allowed, rejecting the rest of the loop if not
        if ind >= len(index_E) or index_E[ind] == -1:
            continue
        # deciding whether to choose the flip
        lng_new = ln_g[index_E[ind]]
        lng_old = ln_g[index_E[E_initial - E_min]]
        diff = lng_old - lng_new
        P = 1.0
        # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
        if diff < 0:
            P = np.exp(diff) * np.exp(-dE / T)
        if np.random.rand() < P:
            s_eq[r, c] = s_new  # accept the flip
            E_initial = E_new
    return s_eq
'''
def observables(M, E):
    n = M.shape[0]
    m = M.reshape(-1, 1) / (num_sites**2)   # normalise to [-1,1]
    e = E.reshape(-1, 1) / (2 * num_sites**2)  # normalise to [-1,1]
    psi = np.hstack([m, m**2, m**3, m*e, e])   # shape (n, 5)
    return psi

def edmd(X1,X2, Y1,Y2):
    psi_X = observables(X1,X2)
    psi_Y = observables(Y1,Y2)
    M = X1.shape[0]  # number of snapshots
    print(psi_X.shape)
    G = (psi_X.T @ psi_X) / M  # edmd gram matrix (covariance matrix)
    A = (psi_X.T @ psi_Y) / M  # edmd matrix
    K = np.linalg.lstsq(G + 1e-6 * np.eye(5), A, rcond=None)[0]  # matrix was ill-conditioned and division by zero errors arose. had to use matrix norm
    eigvals = np.linalg.eigvals(K)
    eigvals = np.sort(np.abs(eigvals))[::-1]  # sorted eigenvalues (descending)
    return eigvals


num_sites = 16  # number of lattice sites
num_steps = int(10e7)  # number of steps of mc algo
J = 1  # ferromagnetic
kb = 1  # boltzman constant in units

#list of all energy values

# discrete time evolution of state of the model using Koopman operator (edmd)

# defining parameter (temperature range)
T = np.linspace(2,3,50)
T_relax=[]
gaps = []
relax = []
gr=np.zeros(len(T))
for k in range(len(T)):
    t = T[k]
    s_2d=random_lattice(num_sites)
    M,E_glauber,gr[k]= glauber(J, s_2d, t)
    # pairs of successive magnetisations
    X1 = M[:-1]
    Y1 = M[1:]
    X2= E_glauber[:-1]
    Y2= E_glauber[1:]
    koop_eigvals = edmd(X1,X2,Y1,Y2)
    spec_gap = 1 - np.abs(koop_eigvals[1])
    gaps.append(spec_gap)
    # finding Koopman operator of discrete-time evolution of DoS
    # time-series pairs such that X[i]=(E[i],M[i]) and Y[i]=(E[i+1],M[i+1]) (state vector)
    # X =np.column_stack([t[:-1],M[:-1]])
    # Y= np.column_stack([t[1:],M[1:]])
    if spec_gap!=0:
        relax.append(1/spec_gap) #relaxation time
        T_relax.append(t)

#plotting spectral gap
gaps_smooth=savgol_filter(gaps, 11, 3)
plt.figure()
plt.plot(T, gaps, label='raw')
plt.plot(T , gaps_smooth, 'r--', label='smoothed')
plt.axvline(x=2 / np.log(1 + np.sqrt(2)),linestyle='--',label='Tc (Onsager)')
plt.xlabel('Temperature')
plt.ylabel('Spectral Gap ')
plt.grid(True)
plt.title('Koopman Spectral Gap vs Temperature')
plt.legend()
plt.show()


#finding Tc
print('-------------------------------\n 16 X 16 LATTICE \n-------------------------------')
min_idx1=np.argmin(gaps)
tc_koop1=T[min_idx1]
err_koop1= np.std(gaps)
print('Using the least value of the Koopman spectral gap on 16 x 16 lattice, Tc = ',tc_koop1,' ± ',err_koop1)
print('Onsager Tc = 2.269')
print('Percentage error in estimate = ', (np.abs(tc_koop1 - 2.269)/2.269)*100,'%')
print('GR statistic for M:')
print(gr)

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
