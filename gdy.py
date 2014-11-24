#
#  GDy.py
#  
#
#  Created by Daniel Citron on 9/06/13.
#   
#  GDy = Graph Dynamics
#  Building upon the code from before (SIRS, SIS), here's a more elegant way to
#   implement disease dynamics on networks with large numbers of nodes
#   This code will work best for large graphs with low density of edges, since
#   it relies on sparse matrixes for best efficiency.


import cPickle as pickle
import networkx as nx
import random, scipy, numpy, sets
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from time import asctime
import scipy.sparse as ssp
from matplotlib.pyplot import pcolormesh

###--------------------------------------------------------------------------------------------------------
# Define the substrate object, which contains the adjacency matrix and node status info
###--------------------------------------------------------------------------------------------------------

class Substrate:

    def __init__(self, G):
        n = len(G)
        self.status = numpy.zeros(n)
        # create matrix representation of adjacency matrix
        if n > 500:
            self.A = nx.to_scipy_sparse_matrix(G)
        else:
            self.A = nx.adjacency_matrix(G)


# Define methods for manipulating the status of individual or groups of nodes
    def GetStatus(self):
        """Return the status of the substrate object"""
        return numpy.copy(self.status)

    
    def SetStatus(self, newstatus):
        """Set the status of the substrate object"""
        if len(self.status) == len(newstatus):
            self.status = newstatus
        else: print 'Error: newstatus has wrong size'


    def InitStatus(self):
        """Reset status (everyone susceptible)"""
        self.SetStatus(numpy.zeros(len(self.status)))


    def InfectN(self, N = 1):
        """ Infect N randomly chosen nodes. """
        nn = random.sample(range(len(self.status)), N)
        for n in nn:
            self.status[n] = 1
    
    
    def InfectGroup(self, nn = [1]):
        """Infect a specific group of nodes """
        for n in nn:
            self.status[n] = 1
    
    
    def SuscN(self, N = 1):
        """ Send N randomly chosen nodes to susceptible. """
        nn = random.sample(range(len(self.status)), N)
        for n in nn:
            self.status[n] = 0
    
    
    def SuscGroup(self, nn = [1]):
        """ Send specific group of nodes to susceptible """
        for n in nn:
            self.status[n] = 0


    def RecoverN(self, N = 1):
        """ Recover N randomly chosen nodes. """
        nn = random.sample(range(len(self.status)), N)
        for n in nn:
            self.status[n] = 2


    def RecoverGroup(self, nn = [1]):
        """ Infect a specific group of nodes """
        for n in nn:
            self.status[n] = 2


# Define methods for measuring the properties of the graph
    def degree(self):
        """Return the degree sequence of the graph"""
        return numpy.array(self.A.sum(0))[0]
        
        
    def Infecteds(self):
        """Return list of infected nodes"""
        return numpy.where(self.status == 1)[0]


    def Susceptibles(self):
        """Return list of susceptible nodes"""
        return numpy.where(self.status == 0)[0]


    def Recovereds(self):
        """Return list of infected nodes"""
        return numpy.where(self.status == 2)[0]
        
        
        
    def DetermS(self, t1 = 0, t2 = 0):
        """
        For deterministic dynamics, return the infected nodes
        """
        return numpy.where(self.status == 0)[0]
         
        
    def DetermI(self, t1, t2):
        """
        For deterministic dynamics, return the infected nodes
        """
        return numpy.where(self.status > t2)[0]
        
        
    def DetermR(self, t1, t2):
        """
        For deterministic dynamics, return te recovered nodes
        """
        return numpy.array(list(set(numpy.where(self.status >0)[0]) & set(numpy.where(self.status <= t2)[0])))
        
        
# Define methods for conducting the dynamics
    def SISupdate(self, beta, gam, nsteps = 1, update = True, ret = False):
        """
        Conduct n updates of status using SIS dynamics
        beta, gam are the SIS dynamics parameters
        update - set to True if we want to actually change the status between time steps
        r - set to True if we want to return the status vector at the end
        """
        nnodes = len(self.status)
        oldstatus = self.GetStatus() # store old status
        for i in range(nsteps):
            r = numpy.random.random(nnodes) # vector of random numbers for each node
            # count the number of infected neighbors, allow for possibility of using sparse array
            if ssp.issparse(self.A): # scipy sparse array
                I = self.A * (self.status == 1)
            else: # dense numpy array
                I = numpy.array(numpy.dot(self.A, self.status == 1))[0]
            newstatus = numpy.copy(self.status) # create an array to store the new status
            newstatus[(self.status == 0) & (r < 1-numpy.exp(-beta*I))] = 1 # condition for susceptible nodes becoming infected
            newstatus[(self.status == 1) & (r < gam)] = 0 # condition for infected nodes becoming susceptible again
            self.SetStatus(newstatus)
        if not update: self.SetStatus(oldstatus) # return the Substrate to the position from the start...
        if ret: return self.status
        
        
    def SIRSupdate(self, beta, gam, rho, nsteps = 1, update = True, ret = False):
        """
        Conduct n updates of status using SIRS dynamics
        beta, gam, rho are the SIS dynamics parameters
        update - set to True if we want to actually change the status between time steps
        ret - set to True if we want to return the status vector at the end
        """
        nnodes = len(self.status)
        oldstatus = self.GetStatus() # store old status
        for i in range(nsteps):
            r = numpy.random.random(nnodes) # vector of random numbers for each node
            # count the number of infected neighbors, allow for possibility of using sparse array
            if ssp.issparse(self.A): # scipy sparse array
                I = self.A * (self.status == 1)
            else: # dense numpy array
                I = numpy.array(numpy.dot(self.A, self.status == 1))[0]
            newstatus = numpy.copy(self.status) # create an array to store the new status
            newstatus[(self.status == 0) & (r < 1-numpy.exp(-beta*I))] = 1 # condition for susceptible nodes becoming infected
            newstatus[(self.status == 1) & (r < gam)] = 2 # condition for infected nodes becoming susceptible again
            newstatus[(self.status == 2) & (r < rho)] = 0 # condition for recovered nodes' waning immunity
            self.SetStatus(newstatus)
        if not update: self.SetStatus(oldstatus) # return the Substrate to the position from the start...
        if ret: return self.status


# Define methods for conducting the dynamics, deterministic recovery
    def PrepareDeterm(self, tau, tau2 = 0):
        """
        Change the status vector to reflect the deterministic dynamics
        After initially infecting using S.InfectN(), 
        run this to set the status of all initial infecteds to tau
        """
        self.status[self.status != 0] = tau + tau2
        
        

    def SISdeterm(self, r, tau, nsteps = 1, update = True, ret = False):
        """
        Conduct n updates of status using SIS dynamics
        beta - is the SIS dynamics parameter
        tau - number of time steps that nodes remain infected
        update - set to True if we want to actually change the status between time steps
        r - set to True if we want to return the status vector at the end
        """
        nnodes = len(self.status)
        oldstatus = self.GetStatus() # store old status
        for i in range(nsteps):
            rv = numpy.random.random(nnodes) # vector of random numbers for each node
            # count the number of infected neighbors, allow for possibility of using sparse array
            if ssp.issparse(self.A): # scipy sparse array
                I = self.A * (self.status > 0)
            else: # dense numpy array
                I = numpy.array(numpy.dot(self.A, self.status > 0))[0]
            newstatus = numpy.copy(self.status) # create an array to store the new status
            newstatus[(self.status == 0) & (rv < r*I)] = tau # condition for susceptible nodes becoming infected
            newstatus[(self.status > 0)] -= 1 # condition for infected nodes becoming susceptible again
            self.SetStatus(newstatus)
        if not update: self.SetStatus(oldstatus) # return the Substrate to the position from the start...
        if ret: return self.status
        
        
        
    def SIRSdeterm(self, r, tau1, tau2, nsteps = 1, update = True, ret = False):
        """
        Conduct n updates of status using SIS dynamics
        beta - is the SIS dynamics parameter
        tau1 - number of time steps that nodes remain infected
        tau2 - number of time steps that nodes remain recovered
        update - set to True if we want to actually change the status between time steps
        r - set to True if we want to return the status vector at the end
        """
        nnodes = len(self.status)
        oldstatus = self.GetStatus() # store old status
        for i in range(nsteps):
            rv = numpy.random.random(nnodes) # vector of random numbers for each node
            # count the number of infected neighbors, allow for possibility of using sparse array
            if ssp.issparse(self.A): # scipy sparse array
                I = self.A * (self.status > tau2)
            else: # dense numpy array
                I = numpy.array(numpy.dot(self.A, self.status > tau2))[0]
            newstatus = numpy.copy(self.status) # create an array to store the new status
            newstatus[(self.status == 0) & (rv < r*I)] = tau1 + tau2 # condition for susceptible nodes becoming infected
            newstatus[(self.status > 0)] -= 1 # condition for infected nodes becoming susceptible again
            self.SetStatus(newstatus)
        if not update: self.SetStatus(oldstatus) # return the Substrate to the position from the start...
        if ret: return self.status


###--------------------------------------------------------------------------------------------------------
# Methods for making measurements of the graph
###--------------------------------------------------------------------------------------------------------

def Khisto(s):
    """
    Degree distribution histogram of network
    s - network substrate
    """
    return Counter(s.degree())


def MeanK(s):
    """
    Return mean degree of network
    s - network substrate
    """
    k = Khisto(s)
    return numpy.sum([1.*i*k[i] for i in k])/len(s.status)
    
    
    
def StdK(s):
    """
    Return mean degree of network
    s - network substrate
    """
    k = Khisto(s)
    return numpy.sum([1.*i*i*k[i] for i in k])/len(s.status)
    


def Pk(s):
    """
    Returns the degree distribution of G as a dictionary = {degree k: probability of k}
    Degree distribution measures probabilities
    """
    pk = {}
    ks = Khisto(s) # {degree: number of nodes with degree}
    for k in ks.keys(): pk[k] = 1.*ks[k]/len(s.status)
    return pk



def RPk(S, ra = True):
    """
    Returns the residual degree distribution as a dictionary: 
    {degree k: number of nodes with residual degree k}
    if ra = 1, return as an array, else return as a dict
    """
    L = set(S.Susceptibles()) | set(S.Infecteds()) # Set of all non-recovered nodes (residual nodes)
    if ssp.issparse(S.A):
        Rdegree = numpy.array([len( L & set(S.A.getrow(i).indices)) for i in L])
        # for each residual node, count the number of connecting residual nodes
    else:
        Rdegree = numpy.array([len(L & set(numpy.where(numpy.array(S.A)[i] > 0)[0])) for i in range(len(S.status))])
    pk = Counter(Rdegree)
    nn = 1.*numpy.sum(pk.values())
    if ra:
        return numpy.array([(k, pk[k]/nn) for k in sorted(pk.keys())]).transpose()
    else:
        return dict([(k, pk[k]/nn) for k in sorted(pk.keys())])



###--------------------------------------------------------------------------------------------------------
# Methods for measuring critical behavior of the SIRS dynamics
###--------------------------------------------------------------------------------------------------------
def SIRS_Compare(G, nobs, gam = .005, R0s = None, rhos = None, mft = True, prev = None, debug = False):
    """
    Measure the SIRS phase diagram properties: I*, <m>, <m^2>, number of connected components of residual network
    Then, extract (one instance of) the residual network and measure SIS I*
    Input   graph G, will be converted to Substrate class object
            nobs - number of observations if endemic state is reached
            gam - single number determines time step of simulation
            R0s - array of parameters: beta*<k>/gam
            rhos - array of parameters: rho/gam
            mft - if true, intelligently set the starting status of the graph
            prev - if not ==None, should be a list of 4 previously written dictionaries
    Output  idict = {(rho, R0) : I*}
            mdict = {(rho, R0) : <m>}
            m2dict = {(rho, R0) : <m^2>}
            ccdict = {(rho, R0) : # connected components}
            sisdict = {(rho, R0) : SIS I*}     
    """
    if R0s == None: R0s = numpy.array(range(45,65))/50.
    if rhos == None: rhos = numpy.logspace(-3, 2, 20)
    S = Substrate(G)
    S.InitStatus()
    meank = MeanK(S)
    if prev == None:
        idict = {}
        mdict = {}
        m2dict = {}
        ccdict = {}
        sisdict = {}
    else: idict, mdict, m2dict, ccdict, sisdict = prev
    for r in rhos: # r is rho/gam, while simulation uses just rho
        for R0 in R0s: # R0 is beta<meank>/gam, while simulation uses beta
            if (r, R0) not in idict:
                beta = R0*gam/meank
                rho = r*gam
                # extract SIRS simulation results
                a,b,c,d = SimulSIRS(S, G, gam, beta, rho, nobs, mft, debug)
                idict[r, R0] = a
                mdict[r, R0] = b
                m2dict[r, R0] = c
                ccdict[r, R0] = d
                # now begin SIS simulation
                resid_nodes = list(set(S.Susceptibles()) | set(S.Infecteds()))
                if debug: print 'Residual network size:',  len(resid_nodes)
                g = nx.subgraph(G, resid_nodes)
                s = Substrate(g)
                s.InfectGroup(range(len(S.Infecteds())))
                sis = SimulSIS(s, g, gam, beta, nobs, debug)
                sisdict[r, R0] = sis
    return idict, mdict, m2dict, ccdict, sisdict
    
    
    
def SIRS_Compare_data(data, total = None):
    """
    The SIRS idict and sisidict data sets are in the form {(rho, R0), array([[S*],[I*]])
    Want to find the fractional average I* over time
    Input:  idict or sisidict, output from SIRS_Compare()
            total - if given as a number, this is the total number of nodes in the graph, appropriate for idict
                    if given as none, just return I/(S + I), appropriate for sisidict
    Output: d = {(rho, R0), I*/N}
    """
    d = {}
    for key in data:
        a = data[key]
        if total == None: 
            d[key] = 1.*numpy.mean(a[1])/numpy.mean(a[1] + a[0])
        else:
            d[key] = 1.*numpy.mean(a[1])/total
    return d
    


def SIRS_Diagram(G, nobs, gam = .005, R0s = None, rhos = None, mft = True, prev = None, debug = False):
    """
    Measure the SIRS phase diagram properties: I*, <m>, <m^2>, number of connected components of residual network
    The number of connected components is calculated each time step with reference to the original graph
    Input   graph G, will be converted to Substrate class object
            nobs - number of observations if endemic state is reached
            gam - single number determines time step of simulation
            R0s - array of parameters: beta*<k>/gam
            rhos - array of parameters: rho/gam
            mft - if true, intelligently set the starting status of the graph
            prev - if not ==None, should be a list of 4 previously written dictionaries
    Output  idict = {(rho, R0) : I*}
            mdict = {(rho, R0) : <m>}
            m2dict = {(rho, R0) : <m^2>}
            ccdict = {(rho, R0) : # connected components}
    """
    if R0s == None: R0s = numpy.array(range(45,65))/50.
    if rhos == None: rhos = numpy.logspace(-3, 2, 20)
    S = Substrate(G)
    S.InitStatus()
    meank = MeanK(S)
    if prev == None:
        idict = {}
        mdict = {}
        m2dict = {}
        ccdict = {}
    else: idict, mdict, m2dict, ccdict = prev
    for r in rhos: # r is rho/gam, while simulation uses just rho
        for R0 in R0s: # R0 is beta<meank>/gam, while simulation uses beta
            if (r, R0) not in idict:
                beta = R0*gam/meank
                rho = r*gam
                a,b,c,d = SimulSIRS(S, G, gam, beta, rho, nobs, mft, debug)
                idict[r, R0] = a
                mdict[r, R0] = b
                m2dict[r, R0] = c
                ccdict[r, R0] = d
    return idict, mdict, m2dict, ccdict



def SimulSIRS(S, G, gam, beta, rho, nobs, mft = True, debug = False):
    """
    Measure the equilibrium endemic state for the given input parameters
    """
    idict = []
    mdict = []
    m2dict = []
    ccdict = []
    meank = MeanK(S)
    stdk = StdK(S)
    nn = len(S.status) # count number of nodes in the network...
    # Infect an initial set of nodes in the network
    S.InitStatus()
    if mft:
        S.InfectGroup(range(max(int(nn*(1-gam/beta/meank)),nn/10)))
    else:
        S.InfectGroup(range(int(nn/2))) # infect the first half of the nodes...    
    # First we equilibrate
    if debug: print 'Begin equilibrating SIRS', beta, rho, asctime()
    for i in range(25):
        S.SIRSupdate(beta, gam, rho, nsteps = int(5./gam))
        # check to see if the network has entered the endemic state or not...
        if len(S.Infecteds()) == 0:
            idict = numpy.array([[nn]*nobs, [0]*nobs])
            mdict = numpy.array([meank]*nobs)
            m2dict = numpy.array([stdk]*nobs)
            ccdict = numpy.array([1]*nobs)
            return idict, mdict, m2dict, ccdict
    # Now we begin making observations:
    if debug: print 'Finished equilibrating, begin SIRS measurements', beta, rho, asctime()
    for i in range(nobs):
        S.SIRSupdate(beta, gam, rho, nsteps = int(5./gam))
        nI = len(S.Infecteds())
        nS = len(S.Susceptibles())
        if nI == 0:
            idict = numpy.array([[nn]*nobs, [0]*nobs])
            mdict = numpy.array([meank]*nobs)
            m2dict = numpy.array([stdk]*nobs)
            ccdict = numpy.array([1]*nobs)
            return idict, mdict, m2dict, ccdict
        rpk = RPk(S, True)
        idict.append(numpy.array([nS, nI]))
        mdict.append(numpy.dot(rpk[0], rpk[1]))
        m2dict.append(numpy.dot(rpk[0]**2, rpk[1]))
        ccdict.append(ConnectedComponents(G, S))
    return numpy.array(idict).transpose(), numpy.array(mdict), numpy.array(m2dict), numpy.array(ccdict)



def SimulSIS(S, G, gam, beta, nobs, debug = False):
    """
    Measure the equilibrium endemic state for the given input parameters
    For now, the substrate passed to the simulation is already infected...
    """
    idict = []
    meank = MeanK(S)
    stdk = StdK(S)
    nn = len(S.status) # count number of nodes in the network...  
    # First we equilibrate
    if debug: print 'Begin equilibrating SIS', beta, asctime()
    for i in range(25):
        S.SISupdate(beta, gam, nsteps = int(5./gam))
        # check to see if the network has entered the endemic state or not...
        if len(S.Infecteds()) == 0:
            idict = numpy.array([[nn]*nobs, [0]*nobs])
            return idict
    # Now we begin making observations:
    if debug: print 'Finished equilibrating, begin SIS measurements', beta, asctime()
    for i in range(nobs):
        S.SISupdate(beta, gam, nsteps = int(5./gam))
        nI = len(S.Infecteds())
        nS = len(S.Susceptibles())
        if nI == 0:
            idict = numpy.array([[nn]*nobs, [0]*nobs])
            return idict
        idict.append(numpy.array([nS, nI]))
    return numpy.array(idict).transpose()



def ConnectedComponents(G, S):
    """
    Find the number of connected components of the residual graph of G
    Residual nodes are found by pulling out all non-recovered nodes 
        and creating a subgraph of G using the list of nodes
    """
    resids = numpy.hstack([S.Infecteds(), S.Susceptibles()])
    g = G.subgraph(resids)
    return len(nx.connected_component_subgraphs(g))



def CP(t, N, k, betas = None, rhos = None, seed = 0):
    #betas = numpy.array(sorted(list(set(numpy.arange(.98,1.1,.01))|set(numpy.arange(1.001,1.005,.001))|set(numpy.arange(1.15,1.5,.05))|set(numpy.arange(1.25,2.25,.25))|set([1.025, 1.015]))))
    #betas = numpy.array(sorted(list(set(numpy.arange(.98,1.1,.01))|set(numpy.arange(1.001,1.005,.001))|set(numpy.arange(1.15,1.5,.05)))))
    #betas = numpy.array(range(5,130,5))/100.
    gam = .005
    if betas == None: betas = numpy.array(range(45,65))/50.
    if rhos == None: rhos = numpy.logspace(-3, 2, 20)
    for i in range(len(rhos)):
        rho = rhos[i]
        exec('f = open("%s_sirs_5k_%d_%d_a_cp.dat", "w")'%(t,k,i))
        data = MeasureCP_SIRS(t, N, k, 10, 100, betas, gam, rho, True, seed, True) # setting resid to be true for now, returning residual degree distributions
        pickle.dump(data[0], f)
        f.close()


def MeasureCP_SIRS(t = 'WS', N = 5000, k = 10, nrun = 1, nobs = 100, betas = None, gam = .005, rho = 1, resid = False, seed = None, debug = False):
    """
    Perform several simulations on random graphs of a specified type
    Each simulation in the loop starts with the same initial set of infected individuals,
        using a different seed to generate the graph.  
        If a seed is specified, the simulation will only run for that seed.
    If a seed is specified, 
    Return a list of dictionaries containing data: [{control parameter = beta*<k>/gam : [data from SimulSIS]}]
    Inputs:
        t - a string that specifies the type of graph: WS, BA, Gnp, delta...
        N - size of the graph (number of nodes)
        k - mean degree
        nrun - number of different graphs to generate...
        nobs - number of observations for each set of parameter values
        betas - numpy array of control parameter values beta*<k>/gam - (pass beta to simulation)
        gam - parameter of SIS simulation, defaults to .005
        rho - parameter of SIS simulation rho/gam - (pass rho to simulation)
        resid - if True, return list of residual degree networks from the endemic state
        seed - integer seed of the random graph.  if specified, run for only a single graph with that random seed
        debug - if True, print debugging messages
    Outputs:
        data - list of outputs from CP_SIS: [{control parameter = beta*<k>/gam : [data from SimulSIS]}]
    """
    data = []
    if nrun == 1 and seed != None: # if we are passing a seed to the program from outside
        print 'Begin constructing graph', seed, asctime()
        if t == 'WS':
            print 'Watts-Strogatz graph'
            G = nx.connected_watts_strogatz_graph(N, k, 1, seed = seed)
        elif t == 'delta':
            print 'Random regular graph'
            G = nx.random_regular_graph(k, N, seed = seed)
        elif t == 'BA':
            print 'Barabas-Albert scale-free graph'
            G = nx.barabasi_albert_graph(N, k/2, seed = seed)
        else:
            print 'Poisson exponential graph'
            G = PoissonGraph(N, 1.*k/N, seed = seed)
        print 'Done constructing graph'
        print 'Begin constructing substrate object', seed, asctime()
        S = Substrate(G) # construct substrate
        del G # clear memory
        S.InitStatus() # allow CP_SIS to infect the graph...
        print 'Begin Critical Point measurement simulation', seed, asctime()
        d = CP_SIRS(S, gam, rho, nobs, betas, True, False, resid, debug)
        print 'Ending simulation on graph', seed, asctime()
        data.append(d)
    else:  # generate random seeds inside the program
        for i in range(nrun):
            print 'Begin constructing graph', i, asctime()
            if t == 'WS':
                print 'Watts-Strogatz graph'
                G = nx.connected_watts_strogatz_graph(N, k, 1, seed = i)
            elif t == 'delta':
                print 'Random regular graph'
                G = nx.random_regular_graph(k, N, seed = i)
            elif t == 'BA':
                print 'Barabas-Albert scale-free graph'
                G = nx.barabasi_albert_graph(N, k/2, seed = i)
            else:
                print 'Poisson exponential graph'
                G = PoissonGraph(N, 1.*k/N, seed = i)
            print 'Done constructing graph'
            print 'Begin constructing substrate object', i, asctime()
            S = Substrate(G) # construct substrate
            del G # clear memory
            S.InitStatus() # allow CP_SIS to infect the graph...
            print 'Begin Critical Point measurement simulation', i, asctime()
            d = CP_SIRS(S, gam, rho, nobs, betas, True, False, resid, debug)
            print 'Ending simulation on graph', i, asctime()
            data.append(d)
    return data



def CP_SIRS(S, gam, rho, nobs, betas = None, mft = True, res = False, resid = False, debug = False):
    """
    Perform a measurement of the critical point for SIS dynamics
    Return a dictionary: {control parameter = beta*<k>/gam : [data from SimulSIS]}
    Inputs:
        S - Substrate type object, already infected
        gam - parameter of simulation 
        rho - parameter of simulation rho/gam- pass rho to simulation
        nobs - number of times to observe the endemic state
        betas - numpy array of control parameter values beta*<k>/gam - pass beta to simulation
        mft - if True, start the simulation with the MFT prediction of endemic infection
            - if False, infect 1/2 of the nodes...
        res - if True, start each simulation (new beta) at the same starting configuration
        debug - if True, print debugging messages
    Outputs:
        data - {control parameter = beta*<k>/gam : [data from SimulSIS]}
    """
    if betas == None: betas = numpy.array(sorted(list(set(range(96, 126, 1))|set(range(130, 200, 10)))))/100.
    N = len(S.status)
    meank = MeanK(S)
    data = {} # dictionary {beta*<k>/gam : [data from SimulSIS]}
    for beta in betas: # for each given beta, perform the simulation
        if len(S.Infecteds()) == 0: # check to see if network is not already infected
            if mft:
                S.InfectGroup(range(max(int(N*(1-1./beta)),100)))
            else:
                S.InfectGroup(range(int(N/2))) # infect the first half of the nodes...    
        if debug: print 'Beta =', beta, asctime()
        if debug: print '%d Infecteds' %len(S.Infecteds())
        d = SimulSIRS(S, beta/meank*gam, gam, rho*gam, nobs, res, resid, debug)
        data[beta] = d
        if mft: #create a new starting configuration (on the next loop)
            S.InitStatus()
    return data




###--------------------------------------------------------------------------------------------------------
#  Same as above, but for deterministic dynamics
###--------------------------------------------------------------------------------------------------------
def Determ_SIRS_Diagram(G, nobs, rs = None, t1 = None, t2s = None, prev = None, debug = False):
    """
    Measure the SIRS phase diagram properties: I*, <m>, <m^2>, number of connected components of residual network
    The number of connected components is calculated each time step with reference to the original graph
    Input   graph G, will be converted to Substrate class object
            nobs - number of observations if endemic state is reached
            rs - list of r parameters to be used in the simulation
            t1 - single number
            t2s - list of t2 parameters to be used in the simulation
            prev - if not ==None, should be a list of 4 previously written dictionaries
            debug - returns messages if True
    Output  idict = {(rho, R0) : I*}
            mdict = {(rho, R0) : <m>}
            m2dict = {(rho, R0) : <m^2>}
            ccdict = {(rho, R0) : # connected components}
    """
    if rs == None: rs = numpy.linspace(.01, .25, 25)
    if t1 == None: t1 = 10
    if t2s == None: t2s = numpy.array(numpy.linspace(0,120, 25), dtype = int)
    S = Substrate(G)
    S.InitStatus()
    meank = MeanK(S)
    if prev == None:
        idict = {}
        mdict = {}
        m2dict = {}
        ccdict = {}
    else: idict, mdict, m2dict, ccdict = prev
    for r in rs: # r is probability of transmission
        for t2 in t2s: # t2 is the time spent not infected
            if (t2, r) not in idict:
                a,b,c,d = Determ_SimulSIRS(S, G, r, t1, t2, nobs, debug)
                idict[t2, r] = a
                mdict[t2, r] = b
                m2dict[t2, r] = c
                ccdict[t2, r] = d
    return idict, mdict, m2dict, ccdict
    



def Determ_SimulSIRS(S, G, r, t1, t2, nobs, debug = False):
    """
    Measure the equilibrium endemic state for the given input parameters
    """
    idict = []
    mdict = []
    m2dict = []
    ccdict = []
    meank = MeanK(S)
    stdk = StdK(S)
    nn = len(S.status) # count number of nodes in the network...
    # Infect an initial set of nodes in the network
    S.InitStatus()
    S.InfectGroup(range(40))
    #S.InfectGroup(range(int(nn/2))) # infect the first half of the nodes...
    S.PrepareDeterm(t1, t2)
    # First we equilibrate
    if debug: print 'Begin equilibrating SIRS', r, t1, t2, asctime()
    for i in range(25):
        S.SIRSdeterm(r, t1, t2, nsteps = 5*(t1+t2))
        # check to see if the network has entered the endemic state or not...
        if len(S.DetermI(t1, t2)) == 0:
            idict = numpy.array([[nn]*nobs, [0]*nobs])
            mdict = numpy.array([meank]*nobs)
            m2dict = numpy.array([stdk]*nobs)
            ccdict = numpy.array([1]*nobs)
            return idict, mdict, m2dict, ccdict
    # Now we begin making observations:
    if debug: print 'Finished equilibrating, begin SIRS measurements', r, t1, t2, asctime()
    for i in range(nobs):
        S.SIRSdeterm(r, t1, t2, nsteps = 5*(t1+t2))
        nI = len(S.DetermI(t1, t2))
        nS = len(S.DetermS(t1, t2))
        if nI == 0:
            idict = numpy.array([[nn]*nobs, [0]*nobs])
            mdict = numpy.array([meank]*nobs)
            m2dict = numpy.array([stdk]*nobs)
            ccdict = numpy.array([1]*nobs)
            return idict, mdict, m2dict, ccdict
        rpk = RPk(S, True)
        idict.append(numpy.array([nS, nI]))
        mdict.append(numpy.dot(rpk[0], rpk[1]))
        m2dict.append(numpy.dot(rpk[0]**2, rpk[1]))
        ccdict.append(ConnectedComponents(G, S))
    return numpy.array(idict).transpose(), numpy.array(mdict), numpy.array(m2dict), numpy.array(ccdict)
    
    
    
def Determ_Rstar(N = 10, nobs = 10, t1 = 10, rs = None, t2s = None, debug = False):
    """
    Run the deterministic dynamics many times to carefully measure the critical point
    N = number of times for each graph
    nobs = number of observations within each run of the simulation
    t1 = tau1, the duration of infection
    rs = list of rs (want between 0 and .1)
    t2s = list of tau2 values, the duration of recovery
    """
    if rs == None: rs = numpy.linspace(.1/50, .1, 50)
    if t2s == None: t2s = range(0,40)
    idict = defaultdict(list)
    mdict = defaultdict(list)
    m2dict = defaultdict(list)
    ccdict = defaultdict(list)
    for r in rs: # r is probability of transmission
        for t2 in t2s: # t2 is the time spent not infected
            if debug: print 't1 = ', t1, 't2 = ', t2, 'r = ', r, asctime()
            for i in range(N): # i is the seed for the random graph
                G = PoissonGraph(14400, 4./14400, seed = i) # Edit this for a different type of random graph
                S = Substrate(G)
                S.InitStatus()
                S.InfectGroup(range(40))
                a,b,c,d = Determ_SimulSIRS(S, G, r, t1, t2, nobs, debug)
                idict[t2, r].append(a)
                mdict[t2, r].append(b)
                m2dict[t2, r].append(c)
                ccdict[t2, r].append(d)
    return idict, mdict, m2dict, ccdict
    # Notes: how to analyze the output from this?
    
    
    
    
def Determ_TimeSeries(S, r, t1, t2, nobs = 100, initn = 40, debug = False):
    """
    S - Substrate object
    r, t1, t2 - parameters of deterministic simulation
    nobs - number of observations to make
    initn - number of initial infecteds
    debug - if True, return debugging messages
    """
    S.InitStatus()
    S.InfectN(initn)
    S.PrepareDeterm(t1, t2)
    data = numpy.array([S.SIRSdeterm(r, t1, t2, update = True, ret = True) for i in range(nobs)])
    return data



def TimeSeries(data, t1, t2):
    """
    Input an array of node status vectors vs. time
    Return three vectors (S[t], I[t], R[t])
    """
    timeseries = numpy.array([[len(numpy.where(data[i] == 0)[0]), #S[t]
        len(numpy.where(data[i] >= t2)[0]), #I[t]
        len(set(numpy.where(data[i] >0)[0]) & set(numpy.where(data[i] <= t2)[0])) #R[t]
        ] for i in range(len(data))])
    return timeseries.transpose()



###--------------------------------------------------------------------------------------------------------
# Methods for measuring critical behavior of the SIS dynamics
###--------------------------------------------------------------------------------------------------------
def MeasureCP_SIS(t = 'WS', N = 50000, k = 6, nrun = 10, nobs = 100, betas = None, gam = .005, debug = False):
    """
    Perform several simulations on random graphs of a specified type
    Return a list of dictionaries containing data: [{control parameter = beta*<k>/gam : [data from SimulSIS]}]
    Inputs:
        t - a string that specifies the type of graph: WS, BA, Gnp, delta...
        N - size of the graph (number of nodes)
        k - mean degree
        nrun - number of different graphs to generate...
        nobs - number of observations for each set of parameter values
        betas - numpy array of control parameter values beta*<k>/gam - pass beta to simulation
        gam - parameter of SIS simulation, defaults to .005
        debug - if True, print debugging messages
    Outputs:
        data - list of outputs from CP_SIS: [{control parameter = beta*<k>/gam : [data from SimulSIS]}]
    """
    data = []
    for i in range(nrun):
        print 'Begin constructing graph', i, asctime()
        if t == 'WS':
            print 'Watts-Strogatz graph'
            G = nx.connected_watts_strogatz_graph(N, k, 1, seed = i)
        elif t == 'delta':
            print 'Random regular graph'
            G = nx.random_regular_graph(k, N, seed = i)
        elif t == 'BA':
            print 'Barabas-Albert scale-free graph'
            G = nx.barabasi_albert_graph(N, k/2, seed = i)
        else:
            print 'Poisson exponential graph'
            G = PoissonGraph(N, 1.*k/N, seed = i)
        print 'Done constructing graph'
        print 'Begin constructing substrate object', i, asctime()
        S = Substrate(G) # construct substrate
        del G # clear memory
        S.InitStatus() # allow CP_SIS to infect the graph...
        print 'Begin Critical Point measurement simulation', i, asctime()
        d = CP_SIS(S, gam, nobs, betas, True, False, debug)
        print 'Ending simulation on graph', i, asctime()
        data.append(d)
    return data
    


def CP_SIS(S, gam, nobs, betas = None, mft = True, res = False, debug = False):
    """
    Perform a measurement of the critical point for SIS dynamics
    Return a dictionary: {control parameter = beta*<k>/gam : [data from SimulSIS]}
    Inputs:
        S - Substrate type object, already infected
        gam - parameter of SIS simulation
        betas - numpy array of control parameter values beta*<k>/gam - pass beta to simulation
        mft - if True, start the simulation with the MFT prediction of endemic infection
            - if False, infect 1/2 of the nodes...
        res - if True, start each simulation (new beta) at the same starting configuration
        debug - if True, print debugging messages
    Outputs:
        data - {control parameter = beta*<k>/gam : [data from SimulSIS]}
    """
    if betas == None: betas = numpy.array(sorted(list(set(range(96, 126, 1))|set(range(130, 200, 10)))))/100.
    N = len(S.status)
    meank = MeanK(S)
    data = {} # dictionary {beta*<k>/gam : [data from SimulSIS]}
    for beta in betas: # for each given beta, perform the simulation
        if len(S.Infecteds()) == 0: # check to see if network is not already infected
            if mft:
                S.InfectGroup(range(int(N*(1-1./beta))))
            else:
                S.InfectGroup(range(int(N/2))) # infect the first half of the nodes...    
        if debug: print 'Beta =', beta, asctime()
        d = SimulSIS(S, beta/meank*gam, gam, nobs, res, debug)
        data[beta] = d
        if mft: #create a new starting configuration (on the next loop)
            S.InitStatus()
    return data
    
    
    
###--------------------------------------------------------------------------------------------------------
### Measuring the simulation results
###--------------------------------------------------------------------------------------------------------

def MeasureIstar(s):
    """ 
    Input the data from a Critical Point measurement
    Return an array: {[betas, mean(I*), std(I*)]} """
    betas = numpy.array(sorted(s.keys()))
    if len(s[betas[0]]) == 1:
        m = numpy.array([(beta, numpy.mean(s[beta]), numpy.std(s[beta])) for beta in betas])
    if len(s[betas[0]]) == 2:
        m = numpy.array([(beta, numpy.mean(s[beta][1]), numpy.std(s[beta][1])) for beta in betas])
    return m.transpose()
    # k = numpy.mean(nx.degree(G).values())
    # n = len(G)
    # plt.errorbar(m[0]*k/gam, m[1]/n, m[2]/n)
    
    
    
def ExtractParameters(idata):
    """
    Input a dictionary, output from DiagramMeasure() or from GDy.SIRS_Diagram simulation
    Dictionary should have the form {(rho, R0) : data }
    Output arrays of values of rho and R0 used in the calculation or simulation
    """
    rhos = numpy.array(sorted(list(set(numpy.array(idata.keys()).transpose()[0]))))
    R0s = numpy.array(sorted(list(set(numpy.array(idata.keys()).transpose()[1]))))
    return rhos, R0s
    
    
    
def Average(idata, index = None):
    """
    Input idata or mdata from DiagramMeasure
    Output a similar dictionary, except with the mean value of idata (not a list)
    
    Index - sometimes the idata will be in the form of a vector: np.array([#Susceptibles, #Infecteds])
        In this case, set index to 1 to extract the number of infecteds...
    """
    new = {}
    if index !=None: #set index = 1 for data on number of infecteds
        for key in idata:
            new[key] = numpy.mean(idata[key][index])
    else:
        for key in idata:
            new[key] = numpy.mean(idata[key])
    return new
    
    
def Fluct(idata, index = None):
    """
    Input idata or mdata from DiagramMeasure
    Output a similar dictionary, except with the size of the fluctuation in the mean (not a list)
    
    Index - sometimes the idata will be in the form of a vector: np.array([#Susceptibles, #Infecteds])
        In this case, set index to 1 to extract the number of infecteds...
    """
    new = {}
    if index !=None: #set index = 1 for data on number of infecteds
        for key in idata:
            h = sorted(list(idata[key][index]))
            # when choosing the 5% and 95% intervals, err on the side of making the interval bigger, not smaller
            new[key] = (h[int(.95*len(h)) - 1] - h[int(.05*len(h)) - 1])/2.
            #new[key] = numpy.std(idata[key][index])
    else:
        for key in idata:
            #new[key] = numpy.std(idata[key])'
            h = sorted(list(idata[key]))
            new[key] = (h[int(.95*len(h)) - 1] - h[int(.05*len(h)) - 1])/2.
    return new
    
    
def FracFluct(idata, index = None):
    """
    Input idata or mdata from DiagramMeasure
    Output a similar dictionary, except with the fractional size of the fluctuation in the mean
    (this is not a list)
    i.e.: std(Idata)/mean(Idata), which gives how far the fluctuation differs from the mean
        empirically speaking, std(Idata) is about 1/2 of the actual size of the fluctuations, 
        the std does not account for extreme values, and it is the extreme values that allow for extinction
    
    Index - sometimes the idata will be in the form of a vector: np.array([#Susceptibles, #Infecteds])
        In this case, set index to 1 to extract the number of infecteds...
    """
    new = {}
    Means = Average(idata, index = index)
    Flucts = Fluct(idata, index = index)
    for key in Means:
        new[key] = Flucts[key]/(max(1., Means[key]))
    return new
    
    
    
def PlotIstar(s, avg = False):
    """
    Input the data output from MeasureCP_SIRS: a list of dicts: [{control parameter = beta*<k>/gam : [data from SimulSIS]}]
    Return list of arrays: [MeasureIstar(s)]
    if avg is True, return a single array of all data sets in s averaged together
    """
    d = []
    for i in s: d.append(MeasureIstar(i))
    if avg: d = sum(numpy.array(d),0)/len(d) # make an array of arrays, sum along the array-stacking axis, and average
    return d
    
    
    
def ColorMap(idata, rhos, R0s, plot = True, Log = True, Ret = False):
    """
    Use pcolormesh to create a heat plot of idata or mdata
    Input:  idata (or mdata) - Output from DiagramMeasure, data to be visualized
            rhos, R0s - numerical values of parameters
            plot - Set to True if we want to visualize
            Log - set to true to put x (rho) axis on a log scale
                This is appropriate, as rho is not zero, and can span large range
            Ret - if true, return the inputs to pcolormesh...
    Output: Plot a picture if plot is set to True
            Output X, Y, C - pcolormesh inputs if Ret input is true
    """
    X, Y = numpy.meshgrid(rhos, R0s)
    C = numpy.array([[idata[rho, R0] for rho in rhos] for R0 in R0s])
    if plot:
        plt.figure()
        pcolormesh(X, Y, C)
        plt.xlabel('rho/gam')
        if Log: 
            plt.xlabel('rho/gam')
            plt.semilogx()
        plt.ylabel('R_0')
        plt.colorbar()
        plt.show()
        # use plt.clim(vmin=, vmax=) to adjust the colorbar scale...)
    if Ret: return X, Y, C
    


def ColorMapM(mdata, m2data, rhos, R0s, meank, plot = True, Log = True, Ret = False):
    """
    Use pcolormesh to create a heat plot of mdata/m2data (ie, <m>/<m^2>
    Input:  mdata, m2data - Output from DiagramMeasure, data to be visualized
            rhos, R0s - numerical values of parameters
            plot - Set to True if we want to visualize
            Log - set to true to put x (rho) axis on a log scale
                This is appropriate, as rho is not zero, and can span large range
            Ret - if true, return the inputs to pcolormesh...
    Output: Plot a picture if plot is set to True
            Output X, Y, C - pcolormesh inputs if Ret input is true
    """
    X, Y = numpy.meshgrid(rhos, R0s)
    C = numpy.array([[meank*mdata[rho, R0]/m2data[rho, R0] for rho in rhos] for R0 in R0s])
    if plot:
        plt.figure()
        pcolormesh(X, Y, C)
        plt.xlabel('rho/gam')
        if Log: 
            plt.xlabel('Log10(rho/gam)')
            plt.semilogx()
        plt.ylabel('R_0')
        plt.colorbar()
        plt.show()
        # use plt.clim(vmin=, vmax=) to adjust the colorbar scale...)
    if Ret: return X, Y, C
    
    
###--------------------------------------------------------------------------------------------------------
### Graph generation
###--------------------------------------------------------------------------------------------------------
def PoissonGraph(N = 1000, p = .005, seed = None):
    """
    Create a Gnp graph, and connect all nodes to the main component
    Initialize the statuses of all nodes, return the graph ready for simulation
    """
    G = nx.fast_gnp_random_graph(N, p, seed)
    Gs = nx.connected_component_subgraphs(G)
    for g in Gs[1:]: # connect all disconnected components...
        node1 = random.choice(Gs[0].nodes())
        node2 = random.choice(g.nodes())
        G.add_edge(node1, node2)
    return G
    
    
def PlaneGraph(l = 100):
    """
    Create a grid graph, with only adjacent neighbors connected
    lxl nodes
    """
    G = nx.Graph()
    for a in range(l*l):
        b = (a+l)%(l*l)
        G.add_edge(a,b)
        G.add_edge(a, (a+1)%l + l*int(a/l))
    return G
    

def PlaneGraphnn(l = 100):
    """
    Create a grid graph
    adjacent and nearest diagonal neighbors connected
    """
    G = nx.Graph()
    for a in range(l*l):
        b = (a+l)%(l*l)
        G.add_edge(a,b)
        G.add_edge(a, (a+1)%l + l*int(a/l))
        G.add_edge(a, (b+1)%l + l*int(b/l))
        G.add_edge(a, (b-1)%l + l*int(b/l))
    return G

###--------------------------------------------------------------------------------------------------------
### Code Graveyard
###--------------------------------------------------------------------------------------------------------
def SimulSIS_OLD(S, beta, gam, nobs, res = True, debug = False):
    """
    Measure the equilibrium endemic state
    Inputs:
        S - Substrate type object, already infected
        beta, gam - parameters of SIS simulation
        nobs - number of observations of the equilibrium state
        res - if True, reset the state of the Substrate S to the original state (before simulation)
        debug - if True, print debugging messages
    Output:
        data - array of I* during the equilibrium endemic state
    """
    data = [] # equilibrium data
    startstate = S.GetStatus()
    # First we equilibrate
    if debug: print 'Begin equilibrating', beta, asctime()
    for i in range(25):
        S.SISupdate(beta, gam, nsteps = int(5./gam))
        #print i, len(S.Infecteds())
        if len(S.Infecteds()) == 0:
            data = [0]*nobs
            if res: S.SetStatus(startstate)
            return numpy.array(data)
    # Now we begin making observations:
    if debug: print 'Finished equilibrating, begin measurements', beta, asctime()
    for i in range(nobs):
        S.SISupdate(beta, gam, nsteps = int(5./gam))
        nI = len(S.Infecteds())
        if nI == 0:
            data = [0]*nobs
            if res: S.SetStatus(startstate)
            return numpy.array(data)
        data.append(nI)
    if res: S.SetStatus(startstate) # restore to starting state if res is True
    if debug: print 'Simulation finished', beta, asctime()
    return numpy.array(data)
    

def SimulSIRS_old(S, beta, gam, rho, nobs, res = True, resid = False, debug = False):
    """
    Measure the equilibrium endemic state
    Inputs:
        S - Substrate type object, already infected
        beta, gam, rho - parameters of SIS simulation
        nobs - number of observations of the equilibrium state
        res - if True, reset the state of the Substrate S to the original state (before simulation)
        resid - if True, return a list of Residual networks...
        debug - if True, print debugging messages
    Output:
        data - array of I* during the equilibrium endemic state
    """
    dataS = [] # Susceptibles equilibrium data
    dataI = [] # Infecteds equilibrium data
    if resid: resid_list = [] # List of residual networks
    startstate = S.GetStatus()
    # First we equilibrate
    if debug: print 'Begin equilibrating', beta, asctime()
    for i in range(25):
        S.SIRSupdate(beta, gam, rho, nsteps = int(5./gam))
        if len(S.Infecteds()) == 0:
            if res: S.SetStatus(startstate)
            return numpy.array([[1]*nobs, [0]*nobs])
    # Now we begin making observations:
    if debug: print 'Finished equilibrating, begin measurements', beta, asctime()
    for i in range(nobs):
        S.SIRSupdate(beta, gam, rho, nsteps = int(5./gam))
        nI = len(S.Infecteds())
        nS = len(S.Susceptibles())
        if nI == 0:
            data = [0]*nobs
            if res: S.SetStatus(startstate)
            return numpy.array([[1]*nobs, [0]*nobs])
        dataS.append(nS)
        dataI.append(nI)
        if resid: resid_list.append(RPk(S, True))
    if res: S.SetStatus(startstate) # restore to starting state if res is True
    if debug: print 'Simulation finished', beta, asctime()
    if resid: return numpy.array([dataS, dataI]), resid_list
    else: return numpy.array([dataS, dataI])