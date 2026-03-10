#using wang-landau mc algo to solve 1d ising model
import random
import numpy as np
import matplotlib.pyplot as plt

num_sites=16 #number of lattice sites
num_steps=int(10e7) #number of steps of mc algo
J=1 #ferromagnetic
kb=1 #boltzman constant in units
s=np.ones(num_sites)
def energy(s,J):
    l=len(s)
    E_i=0
    for i in range (l):
        if i<l-1:
            E_i+=s[i]*s[i+1]
    E_i+=s[l-1]*s[0] #cyclic ising chain
    E_i=-J*E_i
    return E_i

#randomising spins so that net magnetic moment is roughly 0
for i in range(num_sites):
    if random.random()>0.5:
        s[i]=1
    else:
        s[i]=-1

#all possible energies
E=np.array(range(-num_sites,num_sites+1,4),dtype=int) #energies differ by +-4J
index_E=-np.ones(2*num_sites+1,dtype=int) #index of allowed/quantised energies when flipping spins
E_min=min(E)
for i in range(len(E)):
    index_E[E[i]-E_min]=i #looping over negative energies to start from 0

E_initial=int(energy(s,J)) #initial energy of system
M=[sum(s)]
E_plot=[E_initial]
ln_g=np.zeros(len(E)) #log of density of states function
hist=np.zeros(len(E)) #histogram for plotting
f=np.exp(1) #modification factor for W-L
ln_f=1 #ln(f)
#W-L algorithm of random spin flipping
for i in range(num_steps):
    lattice_site=random.randrange(num_sites) #random site
    s_new=s.copy()
    s_new[lattice_site]=s[lattice_site]*(-1) #flipping the spin
    dE= 2*J*s[lattice_site]*(s[(lattice_site-1)%num_sites]+s[(lattice_site+1)%num_sites]) #modulo taken to enforce periodic boundary condition ((num_sites+1)th site=0)
    e_new=E_initial+int(dE) #new value of energy
    #checking to see if energy value is allowed, rejecting the rest of the loop if not
    if index_E[e_new-E_min] ==-1:
        continue

    #deciding whether to choose the flip
    lng_new = ln_g[index_E[e_new - E_min]]
    lng_old = ln_g[index_E[E_initial - E_min]]
    diff=lng_old-lng_new
    # P=1 if gj<=gi and g(Ej)/g(Ei) if gj>gi =e^ln(g(Ej))/e^ln(g(Ei))=e^(ln(g(Ej))-ln(g(Ei)))
    if diff>=0:
        P = 1 #probability of choosing flip
    else:
        P = np.exp(diff)
    if random.random()<P:
        s=s_new.copy() #accept the flip
        E_initial = e_new

    E_plot.append(E_initial)
    hist[index_E[E_initial-E_min]] += 1
    ln_g[index_E[E_initial-E_min]] += ln_f #g->g*f so lng->lng+lnf

    #updating modification factor if histogram flattens after 1000 steps
    if (i+1)%(10*num_sites) == 0 :
        avg=np.mean(hist)
        min_h=min(hist)
        if min_h>avg*(0.9): #measuring how flat we want the histogram to get before we change f
            hist[:]=0
            ln_f=ln_f/2 #modified f (f->f*e^(-2); lnf->lnf/2)
            print("Modifying histogram after n= ",i+1," steps. New modification factor = ",np.exp(ln_f))

#normalising g with condition: g(min)=2 (2 ground states of misaligned spins) and g(max)=2 (2 highest energy states of aligned spins up or down)(g(min)+g(max)=4)(c*g(0)+c*g(N-1)=4 ->c=4/(g0+gN-1)
#normalisation constant log
if ln_g[0]<ln_g[-1]:
    ln_c=np.log(4)-ln_g[-1]-np.log(1+np.exp(ln_g[0]-ln_g[-1]))
else:
    ln_c = np.log(4) - ln_g[0] - np.log(1 + np.exp(ln_g[-1] - ln_g[0]))
ln_g=ln_g+ln_c

#plotting histograms
plt.plot(E,ln_g,'o-')
plt.ylabel('ln(g(E))')
plt.xlabel('E')
plt.title('Log of density of states for N='+str(num_sites)+' spin sites')
plt.grid(True)
plt.show()

T=np.linspace(0.5,5,1000)
Z=np.zeros(len(T)) #Z/Z0 to avoid overflow
U=np.zeros(len(T)) #internal energy
U_2=np.zeros(len(T)) #U**2
C_v=np.zeros(len(T)) #specific heat
#theoretical arrays using transfer matrix method
Z_t=np.zeros(len(T))
C_v_t=np.zeros(len(T))
for k in range(len(T)):
    t=T[k]
    beta=1/(kb*t)
    #taking out max to prevent overflow from numerical recipies
    #z=exp(ln_g(e)-beta*e)
    #z/z0=exp(ln_g(e)-ln_g(e_min) - (beta*e - beta*e_min))
    for i in range(len(E)):
        z=np.exp(ln_g[i]-ln_g[0]-beta*(E[i]-E_min))
        Z[k]+=z
        # using transfer matrix method
        # z=2^n(cosh^n(beta*j)+sinh^n(beta*j))
        lambda1 = 2 * np.cosh(beta * J)
        lambda2 = 2 * np.abs(np.sinh(beta * J))  # abs() for stability
        Z_t[k] = lambda1 ** num_sites + lambda2 ** num_sites
        # internal energy
        U[k]+=z*E[i]
        U_2[k]+=z*(E[i])**2 #sum((g*exp(-beta*e)*e)**2)
    U[k]*=1./Z[k] #U=(sum(g*exp(-beta*e)*e))/Z
    U_2[k]*=1./Z[k]
    C_v[k]=(U_2[k]-(U[k]**2))/(t**2) #Cv = d<E>/dt =(var(U[k])/(t**2)
    C_v_t[k]=num_sites*(J*beta)**2*(1/np.cosh(J*beta))**2

plt.figure(3)
plt.plot(T,C_v,label='numerical')
plt.plot(T,C_v_t,'r:',label='analytical')
plt.ylabel('Cv')
plt.xlabel('T')
plt.legend()
plt.title('Variation of Specific Heat (Cv)  with T for 1D')
plt.grid(True)
plt.show()
