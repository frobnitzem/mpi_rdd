#!/usr/bin/env python3

import numpy as np
from fill import fill

def concat(ans):
    x = []
    for a in ans:
        x.extend(a)
    return x

# Gather together all the partitions, p, from dP : {seq : p}
# with seq % C.procs == C.rank
# seq = 0, 1, ..., N-1
#
# This must be called by all ranks.
def gather_partitions(C, dP, N):
    sets = [[] for i in range(C.procs)] # output sets for each rank
    for seq, p in dP.items():
        j = seq % C.procs
        sets[j].append(p)

    out = [ sets[C.rank] ] # local data skips MPI
    for root in range(C.procs):
        if C.rank == root:
            # ans : [ [partitions belonging to self] ]
            out.extend( C.comm.gather([], root) )
        else:
            C.comm.gather(sets[root], root)

    # re-assemble local partitions
    ans = [ Partition(i,[]) for i in range(C.rank, N, C.procs) ]
    for r in out:
        for p in r:
            ans[p.seq//C.procs].extend(p)
    return ans

def send_chunks(comm, lst, dst, tag, max_elems=100000):
    for i,n in enumerate(range(0, len(lst), max_elems)):
        m = min(n+max_elems, len(lst))
        comm.send(lst[n:m], dest=dst, tag=tag+100*i)

def recv_chunks(comm, lst, off, src, tag, max_elems=100000):
    for i,n in enumerate(range(off, len(lst), max_elems)):
        m = min(n+max_elems, len(lst))
        lst[n:m] = comm.recv(source=src, tag=tag+100*i)

# Use point-to-point send/receives to shuffle list elements between ranks.
#
# C : Context
# sends : [ [(i,j,n)] ] = list of sends at ea. step
# lst : list-like object to index and move around (we add and remove from the end)
def do_sends(C, sends, lst, max_elems=100000):
    for s, sno in enumerate(sends):
        for i,j,n in sno:
            if i == C.rank:
                send_chunks(C.comm, lst[-n:], j, s, max_elems)
                del lst[-n:]
            elif j == C.rank:
                lst.extend( [0]*n )
                recv_chunks(C.comm, lst, len(lst)-n, i, s, max_elems)

class Partition(list):
    def __init__(self, seq, items):
        self.seq = seq
        list.__init__(self, items)
    def filter(self, f):
        return Partition(filter(f, self.seq))
    def map(self, f):
        return Partition(self.seq, [f(x) for x in self])
    def flatMap(self, f):
        ans = []
        for x in self:
            ans.extend(f(x))
        return Partition(self.seq, ans)

    # ans : {f(item) : Partition number f}
    # f : item -> (partition # : int)
    # updates ans with this partition
    def sortby(self, f, ans):
        for x in self:
            k = f(x)
            try:
                ans[k].append(x)
            except KeyError:
                ans[k] = Partition(k, [x])

# An RDD holds the local partitions.
class RDD:
    def __init__(self, ctxt, partitions):
        ctxt.append(self)

        self.C = ctxt
        self.P = partitions

    # TODO: use this in repartition
    # new layout is a single partition per rank
    def foreachPartition(self, f):
        return RDD(self.C, [Partition(self.C.rank, [f(x) for x in self.P])])

    # gather number of partitions to root process
    def getNumPartitions(self):
        return self.C.comm.reduce(len(self.P))

    # Evenly divide the dataset into N partitions.
    def repartition(self, N):
        assert N > 0
        C = self.C
        # jumbo partition with all local items
        lst = Partition(C.rank, concat(self.P))
        items = np.array(C.comm.allgather(len(lst)) ) # items per rank

        total = items.sum()
        tgt = total // N + (np.arange(N) < total%N) # targets for ea. partition

        local_part_sum = np.zeros(C.procs, int) # size sums of local partitions
        for i in range(min(C.procs, N)):
            local_part_sum[i] = tgt[i::C.procs].sum()

        sends = fill(items - local_part_sum)
        # schedule of item sends to achieve len(lst) == local_part_sum[rank]
        do_sends(C, sends, lst)
        assert len(lst) == local_part_sum[C.rank]

        loc = [] # list of local partitions
        n = 0
        for k in range(C.rank, N, C.procs):
            loc.append(Partition(k, lst[n:n+tgt[k]]))
            n += tgt[k]
        assert n == len(lst)
        return RDD(C, loc)

    # Group the values, x, by calc(x) = partition.seq for new partition.
    # Partition holding each data point will be calc(x) % N
    # rank holding each partition will be partition.seq % procs
    def byKey(self, calc, N):
        sends = {}
        for p in self.P:
            p.sortby(lambda x: calc(x) % N, sends)
        # communicate partitions to their ranks
        return RDD(self.C, gather_partitions(self.C, sends, N))

    def filter(self, f):
        return RDD(self.C, [p.filter(f) for p in self.P])

    def map(self, f):
        return RDD(self.C, [p.map(f) for p in self.P])

    def flatMap(self, f):
        return RDD(self.C, [p.flatMap(f) for p in self.P])

    def collect(self):
        ans = self.C.comm.gather(concat(self.P))
        if self.C.rank != 0:
            return ans
        return concat(ans)

    # Turn each element of each partition into its own partition.
    # The old elements are the names of the parquet files to read.
    def partitionFrom(self, f):
        P = []
        for p in self.P:
            for x in p:
                P.append(f(x))
        return RDD(self.C, P)

# create a global context
# The context holds its RDDs
class Context:
    def __init__(self):
        from mpi4py import MPI
        self.rdds = [] # link to all RDDs
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.procs = self.comm.Get_size()

    # Append an RDD to the set of RDDs managed by this context.
    #  Why would we care about saving these things
    #  -- isn't that the user's responsibility?
    def append(self, rdd):
        #self.rdds.append(rdd)
        pass

    # Create an RDD from a sequence of numbers
    # uses a single partition per rank.
    def iterates(self, n):
        return RDD(self, [Partition(self.rank, [i for i in range(self.rank, n, self.procs)])])

########### helpers for pandas ###############
import pandas as pd

# create a partition from a file
def readParquet(fname, idx=True):
    df = pd.read_parquet(fname)
    if idx: # capture the index as a column
        df.reset_index(level=0, inplace=True)
    i = abs(hash(fname))
    return Partition(i, [tuple(r) for r in df.to_numpy()])

