#
#  gdy.py
#  
#
#  Created by Daniel Citron on 9/06/13.
#  Most recent update: 11/24/14 (PEP compliance)
#   
#  gdy = Graph Dynamics
#  
#  This module allows us to implement discrete-time stochastic 
#  simulations of SIR-type disease dynamics.  Most generally,
#  we allow for SIRS dynamics on an arbitrary social network.
#  By properly defining the model parameters, we can simulate
#  SI, SIR, SIS dynamics.
#
#  For large networks (>500 nodes), we use Scipy's sparse matrix
#  methods to perform the simulations.  This has been found to
#  greatly improve the speed of the simulations.  This works best
#  for graphs with a low density of edges.

#  Updates for later:
#  1. Need to be able to properly seed the rng used for updates
#  2. Change certain functions into class methods
#  3. Look at code in graveyard

import cPickle as pickle
import networkx as nx
import random
import scipy
import numpy as np
import sets
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from time import asctime
import scipy.sparse as ssp
from matplotlib.pyplot import pcolormesh


class Substrate:
    """
    Substrate defines a class that keeps track of network connectivity
    which contains the adjacency matrix and node status. Simulations 
    are performed by using methods to operate on a Substrate object.
    These methods look at the adjacency matrix and the status of
    each node in the network.  The adjacency matrix is considered fixed,
    but the statuses of the nodes is what varies over time.
    
    At each discrete time step, SIRS transition probabilities are 
    calculated for each node based on the adjacency matrix, current
    node statuses, and the given model parameters
    
    To create a Substrate object, one simply needs a networkx graph.
    The Substrate object allows easy storage of node status (in a 
    status vector) and defines methods for manipulating nodes' statuses.
    This in turn enables discrete-time stochastic simulations.
    
    Node status...
        = 0 : node is Susceptible to infection
        = 1 : node is Infected, can infect neighbors
        = 2 : node is Recovered, cannot become infected
        
    The model parameters relevant to SIRS-type models are...
        beta_ - beta: per-neighbor rate of infection
        gamma_ - gamma: rate of recovery
        rho_ - rho: rate of loss of immunity ('waning')
    """

    def __init__(self, G):
        """
        Create a Substrate object out of a networkx graph
        Inputs:
            G : networkx graph - defines connectivity between nodes        
        """
        n = len(G)
        # Create vector representation of node statuses
        # Initialize all nodes as "susceptible"
        self.status = np.zeros(n)
        # Create matrix representation of adjacency matrix
        # If the number of nodes > 500, use sparse matrices
        if n > 500:
            self.A = nx.to_scipy_sparse_matrix(G)
        else:
            self.A = nx.adjacency_matrix(G)

###----------------------------------------------------------------------------
#   Methods for manipulating the statuses of nodes
###----------------------------------------------------------------------------

    def getstatus(self):
        """Return the status of the substrate object"""
        return np.copy(self.status)

    def setstatus(self, newstatus):
        """Set the status of the substrate object"""
        if len(self.status) == len(newstatus):
            self.status = newstatus
        else: print 'Error: newstatus has wrong size'

    def initstatus(self):
        """Reset status (everyone susceptible)"""
        self.setstatus(np.zeros(len(self.status)))

    def infect_n(self, N = 1):
        """ Infect N randomly chosen nodes. """
        nn = random.sample(range(len(self.status)), N)
        for n in nn:
            self.status[n] = 1
    
    def infectgroup(self, nn = [1]):
        """Infect a specific group of nodes """
        for n in nn:
            self.status[n] = 1
    
    def susc_n(self, N = 1):
        """ Send N randomly chosen nodes to susceptible. """
        nn = random.sample(range(len(self.status)), N)
        for n in nn:
            self.status[n] = 0
    
    def suscgroup(self, nn = [1]):
        """ Send specific group of nodes to susceptible """
        for n in nn:
            self.status[n] = 0

    def rec_n(self, N = 1):
        """ Recover N randomly chosen nodes. """
        nn = random.sample(range(len(self.status)), N)
        for n in nn:
            self.status[n] = 2

    def recgroup(self, nn = [1]):
        """ Infect a specific group of nodes """
        for n in nn:
            self.status[n] = 2

###----------------------------------------------------------------------------
#   Methods for measuring the properties of the graph
###----------------------------------------------------------------------------

    def degree(self):
        """Return the degree sequence of the graph"""
        return np.array(self.A.sum(0))[0]
        
    def infecteds(self):
        """Return list of infected nodes"""
        return np.where(self.status == 1)[0]
        
    def susceptibles(self):
        """Return list of susceptible nodes"""
        return np.where(self.status == 0)[0]

    def recovereds(self):
        """Return list of infected nodes"""
        return np.where(self.status == 2)[0]
        
###----------------------------------------------------------------------------
#   Methods for implementing discrete-time stochastic simulation
###----------------------------------------------------------------------------        

    def sis_update(self, beta_, gamma_, nsteps = 1, 
                  update = True, ret = False):
        """
        Perform SIS simulation for a fixed number of time steps
        Inputs:
            beta_, gamma_ : SIS dynamics parameters
            nsteps : integer number of time steps
            update : set to True if we want the Substrate's status to be
                     updated after the simulation (usually the case)
            ret    : set to True if we want to return the status vector
                   at the end of the simulation
        Outputs:
            This method returns the final status if ret == True
        """
        nnodes = len(self.status)
        oldstatus = self.getstatus()
        for i in range(nsteps):
            r = np.random.random(nnodes)
            # Count the number of infected neighbors
            if ssp.issparse(self.A): 
                # scipy sparse array
                I = self.A * (self.status == 1)
            else: 
                # dense numpy array
                I = np.array(np.dot(self.A, self.status == 1))[0]
            newstatus = np.copy(self.status)
            # Calculate infection transition probabilities P(S->I)
            newstatus[(self.status == 0) & (r < 1-np.exp(-beta_*I))] = 1 
            # Calculate recovery transition probabilities P(I->S)
            newstatus[(self.status == 1) & (r < gamma_)] = 0
            self.setstatus(newstatus)
        if not update: self.setstatus(oldstatus)
        if ret: return self.status
        
    def sirs_update(self, beta_, gamma_, rho_, nsteps = 1, 
                   update = True, ret = False):
        """
        Perform SIRS simulation for a fixed number of time steps
        Inputs:
            beta_, gamma_, rho_ : SIRS dynamics parameters
            nsteps : integer number of time steps
            update : set to True if we want the Substrate's status to be
                     updated after the simulation (usually the case)
            ret    : set to True if we want to return the status vector
                     at the end of the simulation
        Outputs:
            This method returns the final status if ret == True
        """
        nnodes = len(self.status)
        oldstatus = self.getstatus()
        for i in range(nsteps):
            r = np.random.random(nnodes) 
            # Count the number of infected neighbors
            if ssp.issparse(self.A): 
                # scipy sparse array
                I = self.A * (self.status == 1)
            else: 
                # dense numpy array
                I = np.array(np.dot(self.A, self.status == 1))[0]
            newstatus = np.copy(self.status)
            # Calculate infection transition probabilities P(S->I)
            newstatus[(self.status == 0) & (r < 1-np.exp(-beta_*I))] = 1
            # Calculate recovery transition probabilities P(I->R)
            newstatus[(self.status == 1) & (r < gamma_)] = 2
            # Calculate waning transition probabilities P(R->S)
            newstatus[(self.status == 2) & (r < rho_)] = 0
            self.setstatus(newstatus)
        if not update: self.setstatus(oldstatus)
        if ret: return self.status


###----------------------------------------------------------------------------
# The following functions are ways to make measurements of the 
# underlying network inside a Substrate object.
###----------------------------------------------------------------------------

def degree_dist(s):
    """
    Calculate the degree distribution of a Substrate
    Inputs:
        s : Substrate object
    Outputs:
        Return a Counter histogram {degree:number of nodes with degree}
    """
    return Counter(s.degree())

def mean_degree(s):
    """
    Calculate the mean degree of a Substrate
    Inputs:
        s: Substrate object
    Outputs:
        Return mean degree of Substrate
    """
    ks = degree_dist(s)
    return np.sum([1.*i*ks[i] for i in ks])/len(s.status)
    
def std_degree(s):
    """
    Calculate the second moment of a degree distribution
    Inputs:
        s: Substrate object
    Outputs:
        Return degree standard deviation of Substrate
    """
    ks = degree_dist(s)
    return np.sum([1.*i*i*ks[i] for i in ks])/len(s.status)
    
def p_degree(s):
    """
    Calculate the degree probability distribution of a Substrate
    (The degree probability distribution is normalized to 1.)
    Inputs:
        s  : Substrate object
    Outputs:
        pk : Return a dictionary histogram 
             {degree:probability of nodes with degree}
    """
    pk = {}
    ks = degree_dist(s) # {degree: number of nodes with degree}
    for k in ks.keys(): pk[k] = 1.*ks[k]/len(s.status)
    return pk

def residual_p_degree(s, ra = True):
    """
    Calculate the residual degree distribution of a Substrate.
    'Residual degree' refers to the number of neighbors that are not
    Recovered.  That is to say, count all neighbors that are Susceptible
    or Infected and are still interacting.
    Inputs:
        s  : Substrate object
        ra : If True, returns output as an array
             If False, returns output as a dictionary
    Outputs
        Returns the residual degree distribution.  The format is either
        as an adjacency matrix (if ra == True) or as a dictionary
        {degree: number of nodes with residual degree} (if ra == False)
    """
    # Calculate set of all non-recovered nodes (residual nodes)
    L = set(s.susceptibles()) | set(s.infecteds()) 
    # For each residual node, count the number of connecting residual nodes
    if ssp.issparse(s.A):
        rdegree = np.array([len( L & set(s.A.getrow(i).indices)) for i in L])
    else:
        rdegree = np.array([len(L & set(np.where(np.array(s.A)[i] > 0)[0])) 
                           for i in range(len(s.status))])
    pk = Counter(rdegree)
    nn = 1.*np.sum(pk.values())
    if ra:
        return np.array([(k, pk[k]/nn) for k in sorted(pk.keys())]).transpose()
    else:
        return dict([(k, pk[k]/nn) for k in sorted(pk.keys())])

###--------------------------------------------------------------------
# The following functions are ways to make measurements of the 
# critical behavior of SIRS dynamics.  
#
# We focus on generating SIRS phase diagrams, fixing the underlying 
# connecting graph and gamma. Fixing gamma fixes the time scale of 
# the discrete time simulations to be 1/gamma.  For large gamma, the 
# simulation will progress quickly but will be less accurate.
#
# The phase diagrams are 2-dimensional.  The x-axis is 
# alpha = rho/gamma, the relative rate of waning compared to recovery.
# The y-axis is R0 = beta*<k>/gamma, where <k> is the mean degree.
# R0 is the epidemiological parameter enumerating the average number
# of neighbors infected by an Infected node.
#
# There are a variety of order parameters that the phase diagrams can
# plot on the z-axis that are indicative of the endemic disease state.
# The following functions produce these different order parameters and
# allow for straightforward phase diagram generation and visualization.
###--------------------------------------------------------------------

def sirs_diagram(G, gamma_ = .005, R0s = None, alphas = None, 
                 nobs = 10, mft = True, prev = None, debug = False):
    """
    This script measures SIRS phase diagram properties for a grid
    of parameters (alphas, R0s).  For each combination of grid 
    parameters, we produce a single simulated trajectory and observe
    the system in the quasistationary state after long times. We pick
    a fixed number of times at which we observe the system and measure
    the system at each observation time each time.  The observation 
    times begin after a brief period to allow the transients to die out
    and allow the system to enter into a quasistationary state.  
    Observation times are then spaced out to prevent correlations
    between subsequent observations. Thus, we can measure the 
    quasistationary (endemic disease) state in SIRS simulations.
    
    Phase diagram properties measured:
        I*    : the mean number of infected nodes
        <m>   : the mean residual degree
        <m^2> : the mean-squared residual degree 
                (2nd moment of degree distribution)
        cc    : number of connected components in residual network
    Inputs:
        G     : a networkx graph - use this to define a Substrate object
        gamma_: SIRS model parameter; recovery rate; sets time scale
        R0s   : array/list of R0=beta<k>/gamma values
                y-axis of phase diagram grid
        alphas: array/list of alpha=rho/gamma values
                often these are logarithmically distributed
                x-axis of phase diagram grid
        nobs  : number of observation times, integer
        mft   : If mft==True, begin with a starting configuration of
                Infecteds equal to the mean field theory prediction
                for SIRS, given the parameters (beta, gamma, rho)
                If mft==False, initially infect 1/2 of all nodes
        prev  : Defaults to None. If prev!=None, the function expects
                a list of dictionaries [idict, mdict, m2dict, ccdict]
                representing previous measurements made.  So, one can
                produce an output with this script, then append new
                outputs to the old outputs using prev.
        debug : If debug==True, prints to screen messages indicating
                the simulation's progress.
    Outputs:
        Output is a list of dictionaries [idict, mdict, m2dict, ccdict]
        The dictionaries' keys are (rho, R0) pairs, and values are 
        lists of measurements made at each of the observation times. If
        the quasistationary state has died out, the outputs of idict 
        will all be zero.
        idict : {(rho_, R0) : [S*,I*]}, an array of two vectors
                 representing the (S, I) state at each observation time 
        mdict : {(rho_, R0) : <m>}, mean residual degree vs. time
        m2dict: {(rho_, R0) : <m^2>}, 2nd moment of residual degree            
                distribution vs. time
        ccdict: {(rho_, R0) : #cc}, number of connected components of
                the residual network vs. time
    """
    # default input parameters for phase diagram
    if R0s == None: R0s = np.array(range(45,65))/50.
    if alphas == None: alphas = np.logspace(-3, 2, 20)
    # define Substrate object using the input graph and initiate status
    S = Substrate(G)
    S.initstatus()
    # calculate mean node degree
    meank = mean_degree(S)
    if prev == None:
        idict = {}
        mdict = {}
        m2dict = {}
        ccdict = {}
    else: idict, mdict, m2dict, ccdict = prev
    for alpha in alphas:
        for R0 in R0s:
            if (alpha, R0) not in idict:
                # Define model parameters beta and rho from R0 and alpha
                beta_ = R0*gamma_/meank
                rho_ = alpha*gamma_
                # Perform the simulation
                a,b,c,d = simsirs(S, G, gamma_, beta_, rho_, 
                                    nobs, mft, debug)
                # Append simulation results to the dictionaries
                idict[alpha, R0] = a
                mdict[alpha, R0] = b
                m2dict[alpha, R0] = c
                ccdict[alpha, R0] = d
    return idict, mdict, m2dict, ccdict



def simsirs(S, G, gamma_, beta_, rho_, nobs, mft = True, debug = False):
    """
    Generate a single SIRS trajectory given a connectivity network and
    SIRS model parameters.  Observe the properties of that trajectory
    at a fixed number of observation times.  The observation 
    times begin after a brief period to allow the transients to die out
    and allow the system to enter into a quasistationary state.  
    Observation times are then spaced out to prevent correlations
    between subsequent observations. Thus, we can measure the 
    quasistationary (endemic disease) state in SIRS simulations.
    Inputs:
        S     : a Substrate object created from graph G
        G     : a networkx graph - used to define a Substrate object
        gamma_: SIRS model parameter; recovery rate; sets time scale
        beta_ : SIRS model parameter; per neighbor infection rate
        rho_  : SIRS model parameter; loss of immunity rate
        nobs  : number of observation times; integer
        mft   : If mft==True, begin with a starting configuration of
                Infecteds equal to the mean field theory prediction
                for SIRS, given the parameters (beta, gamma, rho)
                If mft==False, initially infect 1/2 of all nodes
        debug : If debug==True, prints to screen messages indicating
                the simulation's progress.
    Outputs:
        Output is a list of dictionaries [idict, mdict, m2dict, ccdict]
        The dictionaries' keys are (rho, R0) pairs, and values are 
        lists of measurements made at each of the observation times.
        idict : {(rho_, R0) : [S*,I*]}, an array of two vectors
                 representing the (S, I) state at each observation time
        mdict : {(rho_, R0) : <m>}, mean residual degree vs. time
        m2dict: {(rho_, R0) : <m^2>}, 2nd moment of residual degree            
                distribution vs. time
        ccdict: {(rho_, R0) : #cc}, number of connected components of
                the residual network vs. time
    """
    # Initialize empty lists to store simulation results
    idict = []
    mdict = []
    m2dict = []
    ccdict = []
    # Calculate network degree distribution properties
    meank = mean_degree(S)
    stdk = std_degree(S)
    nn = len(S.status)
    # Infect an initial set of nodes in the network
    S.initstatus()
    if mft:
        S.infectgroup(range(max(int(nn*(1-gamma_/beta_/meank)),nn/10)))
    else:
        S.infectgroup(range(int(nn/2)))   
    # First we equilibrate
    if debug: 
        print 'Begin equilibrating SIRS', beta_, rho_, asctime()
    for i in range(25):
        S.sirs_update(beta_, gamma_, rho_, nsteps = int(5./gamma_))
        # Check to see that the quasistationary state has not yet died
        if len(S.infecteds()) == 0:
            idict = np.array([[nn]*nobs, [0]*nobs])
            mdict = np.array([meank]*nobs)
            m2dict = np.array([stdk]*nobs)
            ccdict = np.array([1]*nobs)
            return idict, mdict, m2dict, ccdict
    # Begin making observations of the quasistationary state
    if debug: 
        print 'Finished equilibrating, begin SIRS measurements', \
        beta_, rho_, asctime()
    for i in range(nobs):
        S.sirs_update(beta_, gamma_, rho_, nsteps = int(5./gamma_))
        nI = len(S.infecteds())
        nS = len(S.susceptibles())
        # Check to see that the quasistationary state has not yet died
        if nI == 0:
            idict = np.array([[nn]*nobs, [0]*nobs])
            mdict = np.array([meank]*nobs)
            m2dict = np.array([stdk]*nobs)
            ccdict = np.array([1]*nobs)
            return idict, mdict, m2dict, ccdict
        rpk = residual_p_degree(S, True)
        idict.append(np.array([nS, nI]))
        mdict.append(np.dot(rpk[0], rpk[1]))
        m2dict.append(np.dot(rpk[0]**2, rpk[1]))
        ccdict.append(connectedcomponents(G, S))
    return np.array(idict).transpose(), np.array(mdict), \
           np.array(m2dict), np.array(ccdict)

def connectedcomponents(g, s):
    """
    Find the number of connected components of the residual graph of G.
    The residual graph is found by identifying all Susceptible and 
    Infected nodes and constructing the subgraph of G. It is then
    straightforward to count the number of connected components.
    Inputs:
        G : connectivity graph; networkx undirected Graph
        S : Substrate object created from G; contains
            status vector for identifying residual graph
    Outputs:
        Number of residual graph connected components; integer
    """
    resids = np.hstack([s.infecteds(), s.susceptibles()])
    gsub = g.subgraph(resids)
    return len(nx.connected_component_subgraphs(gsub))
    
###----------------------------------------------------------------------------
### Measuring the simulation results
###----------------------------------------------------------------------------

def data_params(idata):
    """
    Given the output from sirs_diagram(), return two lists of all 
    parameters alphas and R0s.  The input is a single one of the 
    outputs (such as idata or mdata) with the form {(alpha, R0):data}
    Inputs:
        idata : dictionary from output of sirs_diagram()
    Outputs:
        alphas: array vector of alpha=rho/gamma
        R0S   : array vector of R0<k>/gamma
    """
    alphas = np.array(sorted(list(set(np.array(idata.keys()).transpose()[0]))))
    R0s = np.array(sorted(list(set(np.array(idata.keys()).transpose()[1]))))
    return alphas, R0s

def avg_idata(idata, index = None):
    """
    Given the output from sirs_diagram(), perform time averaging on
    the vector of observed values.
    Inputs:
        idata : Dictionary from output of sirs_diagram
        index : Defaults to 0 for mdata, m2data, ccdata. 
                If idata is the dictionary of (S, I) value pairs,
                set index=0 for the number of Susceptibles, 
                set index=1 for the number of Infecteds.
    Outputs:
        new   : A dictionary of all averaged values, given the original 
                time series of observations {(alpha, R0):averaged value}
    """
    new = {}
    if index != None:
        for key in idata:
            new[key] = np.mean(idata[key][index])
    else:
        for key in idata:
            new[key] = np.mean(idata[key])
    return new
    
def fluct(idata, index = None):
    """
    Given the output from sirs_diagram(), measure the time-averaged 
    fluctuation about the mean.  The fluctuation is defined as the
    interval that contains 90% of the data
    Inputs:
        idata : Dictionary from output of sirs_diagram
        index : Defaults to 0 for mdata, m2data, ccdata. 
                If idata is the dictionary of (S, I) value pairs,
                set index=0 for the number of Susceptibles, 
                set index=1 for the number of Infecteds.
    Outputs:
        new   : A dictionary of all fluctuation sizes, given the original 
                time series of observations {(alpha, R0):averaged value}
    """
    new = {}
    if index != None:
        for key in idata:
            h = sorted(list(idata[key][index]))
            new[key] = np.abs((h[int(.05*len(h))-1]-h[int(.95*len(h))-1])/2.)
    else:
        for key in idata:
            h = sorted(list(idata[key]))
            new[key] = np.abs((h[int(.05*len(h))-1]-h[int(.95*len(h))-1])/2.)
    return new
    
def frac_fluct(idata, index = None):
    """
    Given the output from sirs_diagram(), measure the fractional size
    of the time-averaged fluctuation about the mean fluct/mean.
    Inputs:
        idata : Dictionary from output of sirs_diagram
        index : Defaults to 0 for mdata, m2data, ccdata. 
                If idata is the dictionary of (S, I) value pairs,
                set index=0 for the number of Susceptibles, 
                set index=1 for the number of Infecteds.
    Outputs:
        new   : A dictionary of all fractional fluctuation sizes, 
                given the original time series of observations 
                {(alpha, R0):averaged value}
    """
    new = {}
    means = avg_idata(idata, index = index)
    flucts = fluct(idata, index = index)
    for key in means:
        new[key] = flucts[key]/(max(1., means[key]))
    return new
    
def colormap(data, alphas, R0s,
             p = True, logx = True, ret = False):
    """
    Use pcolormesh from matplotlib to create a visualization of the 
    phase diagram in the form of a heat map.
    Input:  
        data  : Dictionary of the form {(alpha, R0}:value}, where
                the value represents <I*>, <S*>, etc.  This dictionary
                is produced by avg_idata(), fluct(), or frac_fluct()
        alphas: Phase diagram parameters on x-axis; rho/gamma
        R0s   : Phase diagram parameters on y-axis; beta<k>/gamma
                These can be obtained with data_params()
        p     : Create a plot of the data if p==True
        logx  : Plot with a logarithmic x-axis (alphas logarithmic)
        ret   : If ret==True, return the values used to create the
                heat map, so that they may be manipulated in the 
                interpreter or elsewhere; defaults to False
    Output:
        Creates a matplotlib heatmap if p==True
        (X, Y, C): These are three arrays that pcolormesh() takes as
                arguments.  These are returned only if ret==True
    """
    X, Y = np.meshgrid(alphas, R0s)
    C = np.array([[data[rho_, R0] for rho_ in alphas] for R0 in R0s])
    if p:
        plt.figure()
        pcolormesh(X, Y, C)
        plt.xlabel('rho/gamma')
        if logx: 
            plt.semilogx()
        plt.ylabel('R_0')
        plt.colorbar()
        plt.show()
    if ret: return X, Y, C
    
###----------------------------------------------------------------------------
# Methods for generating different types of graphs
# The graphs can then be turned into Substrate objects.
###----------------------------------------------------------------------------

def poisson_graph(n = 1000, p = .005, seed = None):
    """
    This function uses networkx's function to generate a Gnp-type
    graph with a Poisson-distributed degree probability distribution.
    Occasionally, however, the Gnp-type graph will have multiple 
    disconnected components: this function connects those components 
    together with edges so that the output is a single connected component.
    Inputs:
        n : number of nodes
        p : edge probability
        seed : seed for random graph generation
    Outputs:
        g : Connected Gnp-type graph, networkx undirected graph object
    """
    g = nx.fast_gnp_random_graph(n, p, seed)
    gs = nx.connected_component_subgraphs(g)
    for subg in gs[1:]: # connect all disconnected components...
        node1 = random.choice(gs[0].nodes())
        node2 = random.choice(subg.nodes())
        g.add_edge(node1, node2)
    return g
    
def plane_graph(L = 100):
    """
    This function defines an LxL grid of nodes and connects all
    nodes to their nearest neighbors (square lattice).  This
    square lattice of nodes has periodic boundary conditions (torus)
    Inputs:
        L : number of nodes on each edge of the grid
    Outputs:
        g : grid graph, networkx undirected graph object
    """
    g = nx.Graph()
    for a in range(L*L):
        b = (a+L)%(L*L)
        g.add_edge(a,b)
        g.add_edge(a, (a+1)%L + L*int(a/L))
    return g
    
def plane_graph_nn(L = 100):
    """
    This function defines an LxL grid of nodes and connects all
    nodes to their nearest neighbors as well as their next-nearest
    neighbors.  Each node is therefore locally connected to 8 others.
    This grid of nodes has periodic boundary conditions (torus)
    Inputs:
        L : number of nodes on each edge of the grid
    Outputs:
        g : grid graph, networkx undirected graph object
    """
    g = nx.Graph()
    for a in range(L*L):
        b = (a+L)%(L*L)
        g.add_edge(a,b)
        g.add_edge(a, (a+1)%L + L*int(a/L))
        g.add_edge(a, (b+1)%L + L*int(b/L))
        g.add_edge(a, (b-1)%L + L*int(b/L))
    return g
    
###----------------------------------------------------------------------------
# Some extra code, still in development
###----------------------------------------------------------------------------

#def SimulSIS(S, G, gamma_, beta_, nobs, debug = False):
#    """
#    Measure the equilibrium endemic state for the given input parameters
#    For now, the substrate passed to the simulation is already infected...
#    """
#    idict = []
#    meank = mean_degree(S)
#    stdk = std_degree(S)
#    nn = len(S.status) # count number of nodes in the network...  
#    # First we equilibrate
#    if debug: print 'Begin equilibrating SIS', beta_, asctime()
#    for i in range(25):
#        S.sis_update(beta_, gamma_, nsteps = int(5./gamma_))
#        # check to see if the network has entered the endemic state or not...
#        if len(S.infecteds()) == 0:
#            idict = np.array([[nn]*nobs, [0]*nobs])
#            return idict
#    # Now we begin making observations:
#    if debug: 
#        print 'Finished equilibrating, begin SIS measurements', 
#                beta_, asctime()
#    for i in range(nobs):
#        S.sis_update(beta_, gamma_, nsteps = int(5./gamma_))
#        nI = len(S.infecteds())
#        nS = len(S.susceptibles())
#        if nI == 0:
#            idict = np.array([[nn]*nobs, [0]*nobs])
#            return idict
#        idict.append(np.array([nS, nI]))
#    return np.array(idict).transpose()

#def ColorMapM(mdata, m2data, alphas, R0s, meank, 
#              p = True, logx = True, ret = False):
#    """
#    Use pcolormesh from matplotlib to create a visualization of the 
#    phase diagram of the 
#    
#    in the form of a heat map.
#    Use pcolormesh to create a heat plot of mdata/m2data (ie, <m>/<m^2>
#    Input:  mdata, m2data - Output from DiagramMeasure, data to be visualized
#            alphas, R0s - numerical values of parameters
#            plot - Set to True if we want to visualize
#            Log - set to true to put x (rho) axis on a log scale
#             This is appropriate, as rho is not zero, and can span large range
#            Ret - if true, return the inputs to pcolormesh...
#    Output: Plot a picture if plot is set to True
#            Output X, Y, C - pcolormesh inputs if Ret input is true
#    """
#    X, Y = np.meshgrid(alphas, R0s)
#    C = np.array([[meank*mdata[rho_, R0]/m2data[rho_, R0] 
#                for rho_ in alphas] for R0 in R0s])
#    if plot:
#       plt.figure()
#        pcolormesh(X, Y, C)
#        plt.xlabel('rho/gamma')
#        if Log: 
#            plt.xlabel('Log10(rho/gamma)')
#            plt.semilogx()
#        plt.ylabel('R_0')
#        plt.colorbar()
#        plt.show()
#        # use plt.clim(vmin=, vmax=) to adjust the colorbar scale...)
#    if Ret: return X, Y, C