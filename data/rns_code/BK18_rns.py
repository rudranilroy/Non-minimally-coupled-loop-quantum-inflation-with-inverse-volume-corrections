# BICEP/Keck Array Nov 2021 Data Products
# BICEP/Keck XIII: Improved Constraints on Primordial Gravitational Waves using Planck, WMAP, and BICEP/Keck Observations through the 2018 Observing Season
# http://bicepkeck.org/
#
# File: BK18_rns.py
# Date: 2021-11-08
#
# This python script, together with the accompanying text files, plots the r-ns 
# constraints and selected inflation models shown in Figure 5 of BK-XIII. 
# It does not require BK18 CosmoMC chains.

import numpy as np
import matplotlib.pyplot as plt

def N_r_ns(r, ns):
    return (r - 16) / (8. * ns - 8 + r) / 2.

def r_ns(ns, p):
    return 8. * (1 - ns) * p / (2. + p)

def ns_N(N, p):  # first order
    return (4 * N - p - 4) / (4. * N + p)

# Show plots
plt.ion()
plt.figure(figsize=(6,6))

# Colors, alphas and labels
colors = [('#9ACB9A','g'), ('#8CD3F5', '#006FED')]
alphas = [1, 0.8]
labels = ['Planck TT,TE,EE+lowE+lensing',
          '                             +BK18+BAO']

# Load and plot Planck 2018 and BK18 contours
for i, filename in enumerate(['P18','BK18']):
  xy = np.loadtxt(filename+'density_xy.txt')
  P = np.loadtxt(filename+'density_P.txt')
  levels = np.loadtxt(filename+'density_levels.txt')
  x = xy[:,0]
  y = xy[:,1]
  plt.contourf(x, y, P, levels, colors=colors[i],alpha=alphas[i])
  plt.contour(x, y, P, levels[:1], colors=colors[i][1], alpha=alphas[i])

plt.ylabel('$r_{0.002}$')
plt.xlabel('$n_s$')
plt.legend(labels,frameon=False,
           labelcolor=[colors[0][1],colors[1][1]])
plt.xlim([0.945, 1])
plt.ylim([0, 0.26])
plt.gca().set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25])
plt.gca().set_xticks([0.95, 0.96, 0.97, 0.98, 0.99, 1.0])

###########################################################################
# Inflation Models
#
# Get monomial inflation filled contours
pl = np.logspace(np.log10(0.001),np.log10(5),1000)
for i in np.arange(0,1000,1):
    p = pl[i]
    ns = np.arange(0.93, 1.1, 0.0001) 
    ns = np.arange(ns_N(50, p), ns_N(60, p), 0.0001)  #
    plt.plot(ns, r_ns(ns, p), color='orange', lw=0.5, alpha=0.25)

# Get Natural Inflation
xvals = np.linspace(0.945,ns_N(60,2))
xn = np.arange(.0001,.1,.0001)
N=50;
n = 2*N
ep = xn/(np.exp(n*xn)-1.)
et = ep - xn
x1 = 1. - 6*ep + 2*et
y1  = 16*ep
plt.plot(x1,y1,color='k',lw=0.3)

N=60;
n = 2*N
ep = xn/(np.exp(n*xn)-1.)
et = ep - xn
x2 = 1. - 6*ep + 2*et
y2  = 16*ep
plt.plot(x2,y2,color='k',lw=0.3)

for i in np.arange(0,250,1):
    plt.plot([x1[i],x2[i]],[y1[i],y2[i]],color='purple',alpha=0.3,lw=0.5)
ns = np.arange(0.91, 1.02, 0.0005)
r = np.arange(0, 0.34, 0.002)
ns, r = np.meshgrid(ns, r)

# this is the first order result
N = N_r_ns(r, ns)
N[r > 8 * (1 - ns)] = 100

# Plot dashed monomial lines for N=50 and N=60
plt.contour(ns, r, N, origin='lower', levels=[50, 60], colors='k', linestyles=':', linewidths=0.3,
           extent=[0.94, 1.01, 0.001, 0.25])

# Get the N=50 and N=60 text
for x, y, lab in zip([0.954, 0.9585], [0.2, 0.211], ['N=50', 'N=60']):
    plt.text(x, y, lab, size=7, rotation=-62, color='k',
             ha="center", va="center", bbox=dict(ec='1', fc='1', alpha=0))

# Convex/Concave line
ns = np.arange(0.9, 1.1, 0.0001)
plt.plot(ns, r_ns(ns, 1), ls='-', color='k', lw=1, alpha=0.8)
# Text for the line
plt.text(0.954, 0.13, 'Convex', size=7, rotation=-30, color='k',
         ha="center", va="center")
plt.text(0.954, 0.116, 'Concave', size=7, rotation=-30, color='k',
         ha="center", va="center")

# Phi^2/3, Phi and Phi^2 labels
modcol = 'red'
plt.text(0.962, 0.155, r'$\phi^2$', fontsize=9, color=modcol, bbox=dict(ec='1', fc='1', alpha=0.8), zorder=-2)
plt.text(0.971, 0.081, r'$\phi$', fontsize=9, color=modcol)
plt.text(0.978, 0.033, r'$\phi^{2/3}$', fontsize=9, color=modcol)

# Get Phi^2/3, Phi and Phi^2 red lines
for p in [2/3, 1, 2]:
    ns = np.arange(0.93, 1.1, 0.0001)

    if p != 1: plt.plot(ns, r_ns(ns, p), ls='-', color='black', lw=0.6, alpha=0.4)
    # print p, ns_N(50, p), ns_N(60, p)
    ns = np.arange(ns_N(50, p), ns_N(60, p), 0.0001)
    plt.plot(ns, r_ns(ns, p), ls='-', color=modcol, lw=1.2, alpha=1)

plt.savefig('BK18_rns.png')
