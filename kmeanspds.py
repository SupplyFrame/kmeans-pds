__author__ = 'ltorres'

import random
import time
from pandas import DataFrame
from multiprocessing import Pool
from distance import *

def dpart(distfun, d, v1, v2):
    """
    Partial Distance function.

    Parameters
    ----------
    distfun : lambda(v1, v2)
            Distance function to use.
    d : integer
            Dimension
    v1, v2 : Series objects
            Vectors to measure distance for.

    Returns
    -------
    out : double
        Distance between x1 and x2.
    """
    both_available = map(lambda(x,y): x and y, zip(v1.notnull(), v2.notnull()))
    b = d - sum(both_available)
    dist = distfun(v1, v2)

    return dist * d / (d - b)

def distance_worker(kmeans, chunk):
    """
    Worker to pre-compute distances in parallel chunks.

    Parameters
    ----------
    kmeans : KMeans instance
           We need the distance function ,dimension and data from the K-Means object.
    chunk : List of Series
            Chunk of data points to calculate distances for.

    Returns
    -------
    out : dictionary of dictionaries
        Distances between data in chunk and all other data elements.  Keys are the series name.
    """
    distances = {}
    count = 0
    for i, x1 in enumerate(chunk):
        distances[x1.name] = {}
        for j, x2 in kmeans.data.iterrows():
            distances[x1.name][x2.name] = dpart(kmeans.distfun, kmeans.d, x1, x2)
            count += 1
    print "chunk finished: %i distances computed" % count
    return distances

def assign_worker(kmeans, chunk):
    """
    Parallel component of assign().  We can allocate points to their closest cluster in parallel.  The points are
    divided into a number of chunks equal to the number of worker threads.  Each thread will get one invocation of
    this method with one chunk.

    Returns
    -------
    out : k lists, each with a list of points assigned to the k'th cluster
    """
    results = [[] for _ in range(kmeans.k)]
    for x in chunk:
        selected_cluster = None
        smallest_distance = None

        for k in range(kmeans.k):
            dist = kmeans.distances[kmeans.clusters[k].mu.name][x.name]
            if smallest_distance is None or dist < smallest_distance:
                smallest_distance = dist
                selected_cluster = k

        results[selected_cluster].append(x)

    return results


class KMeans:
    """
    Implementation of KMeans with Partial Distance Strategy.  Distance and mean functions can be supplied for
    non-euclidean spaces.
    """

    def __init__(self, k, data, distfun=dist_euclidean, mufun=mu_euclidean, epsilon=0.000001, convergence=7, maxiter=20):
        self.k = k
        self.data = data
        self.distfun = distfun
        self.mufun = mufun
        self.epsilon = epsilon
        self.convergence = convergence
        self.maxiter = maxiter
        self.duration = time.time()

        self.N = len(self.data)
        self.d = len(self.data.iloc[0])
        self.clusters = [Cluster(i) for i in range(self.k)]

        self.iterations = 0
        self.converge_history = [[] for _ in range(self.k)]

        self.datachunks = [[] for _ in range(WORKERS)]
        for i, r in self.data.iterrows():
            self.datachunks[i % WORKERS].append(r)

        self.init_centroids()
        self.init_distances()
        pass # done with init

    def init_centroids(self):
        """
        This implementation chooses k random points from the data set and uses them as the initial centroids.
        """
        i = 0
        while i < self.k:
            x = self.data.iloc[random.randint(0, self.N - 1)]
            if all(x.isnull()): continue
            self.clusters[i].mu = x
            i += 1

    def init_distances(self):
        print "Computing distances..."

        self.distances = {}
        for job in [POOL.apply_async(distance_worker, (self, chunk, )) for chunk in self.datachunks]:
            self.distances.update( job.get(timeout=None) )

        print "Distances finished in %is" % (time.time() - self.duration)

    def clear_clusters(self):
        for cluster in self.clusters:
            cluster.clear()

    def assign(self):
        """
        Assign data points to closest cluster
        """
        self.clear_clusters()
        jobs = [POOL.apply_async(assign_worker, (self, chunk,)) for chunk in self.datachunks]
        for job in jobs:
            results = job.get(timeout=None)
            for k, result in enumerate(results):
                self.clusters[k].data += result

    def recenter(self):
        """
        Recalculate the centroids of the clusters
        """
        jobs = [POOL.apply_async(self.mufun, (k, cluster, self.d,)) for k, cluster in enumerate(self.clusters)]
        for job in jobs:
            (k, mu) = job.get(timeout=None)
            self.clusters[k].mu = mu

    def run(self):
        """
        Main loop of the k-means algorithm
        """
        while True:
            t = time.time()
            old_mu = [c.mu for c in self.clusters]

            self.assign()
            self.recenter()

            self.iterations += 1
            print "Iteration %i finished in %is" % (self.iterations, time.time() - t)

            if self.converged(old_mu):
                break

        self.duration = time.time() - self.duration
        return self.output()

    def converged(self, old_mu):
        """
        Checks if the iteration converged.  Convergence is satisfied by any of the following:

          *  If number of iterations > maxiter
          *  if distance(old_mu_k, new_mu_k) < epsilon for all clusters k
          *  If at least n=convergence iterations has passed and mu_k has been cycling between one,
             two or three points for the last n steps.  If all mu_k have been cycling like this, we should quit.
        """
        if self.iterations > self.maxiter:
            self.termination = "Max iterations reached"
            return True

        conv = [False] * self.k
        cyclic = [False] * self.k
        for i, (mu1, mu2) in enumerate(zip(old_mu, [c.mu for c in self.clusters])):
            #
            # If cluster has no data and no mu, consider it converged
            if mu1 is None or mu2 is None:
                conv[i] = True
                continue

            dist = self.distances[mu1.name][mu2.name]
            self.converge_history[i].append(dist)

            if self.iterations > self.convergence and len(set(self.converge_history[i][-(self.convergence):])) <= 3:
                cyclic[i] = True

            conv[i] = dist < self.epsilon

        if all(conv):
            self.termination = "Epsilon %f" % (self.epsilon)
            return True

        if all(cyclic):
            self.termination = "Cycle detected"
            return True

        return False

    def output(self):
        """
        Updates the input data to include the cluster, whether the point is centroid, and how far the point is from
        the centroid.
        """
        self.data['cluster'] = None
        self.data['centroid'] = None
        self.data['mu_dist'] = None

        for cluster in self.clusters:
            for x in cluster.data:
                y = self.data.loc[x.name]
                y['cluster'] = "C" + repr(cluster.i)
                if cluster.mu.name == y.name:
                    y['centroid'] = True
                    y['mu_dist'] = 0
                else:
                    y['mu_dist'] = self.distances[y.name][cluster.mu.name]
                    y['centroid'] = False
                self.data.loc[x.name] = y

        return self.data

    def __repr__(self):
        s = "[KMeans k=%i, iterations=%s, duraiton=%i, termination=%s]" % (self.k, self.iterations, self.duration, self.termination)
        for cluster in self.clusters:
            s += "\n\t" + cluster.__repr__()
        return s

    def __str__(self):
        return self.__repr__()


class Cluster:
    """
    Container for a cluster number k, the centroid point mu, and a list of data assigned to the cluster
    """
    def __init__(self, i):
        self.i = i
        self.mu = None
        self.data = []

    def clear(self):
        del self.data[:]

    def __repr__(self):
        return "[Cluster %i: elements=%s, centroid=%s]" % (self.i, len(self.data), self.mu.name)

    def __str__(self):
        return self.__repr__()


WORKERS = 4
POOL = Pool(processes=WORKERS)