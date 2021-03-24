#!/usr/bin/env python3

from rdd import *

def test_iterate(N):
    C = Context()
    r = C.iterates(N).map(lambda u: (u+12.9,1,2))
    ans = r.getNumPartitions()
    if C.rank == 0: assert ans == C.procs

    pl = r.foreachPartition(len)
    assert isinstance(pl, RDD)

    ans = pl.getNumPartitions()
    if C.rank == 0: assert ans == C.procs
    ans = pl.collect()
    if C.rank == 0: assert isinstance(ans, list)
    if C.rank == 0: assert len(ans) == C.procs

def test_parquet(M = 20):
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        fmt = os.path.join(tmpdir, "data.%d.pq")
        for i in range(M):
            pd.DataFrame([(4*i+j,j*j) for j in range(4)], columns=['A', 'B']).to_parquet(fmt % i)

        C = Context()

        z = C.iterates(M).map(lambda i: fmt%i).partitionFrom(readParquet)
        ans = z.getNumPartitions()
        if C.rank == 0: assert ans == M
        ans = z.foreachPartition(len).collect()
        if C.rank == 0: assert len(ans) == M
        if C.rank == 0: assert np.all(np.array(ans) == ans[0])

def test_repartition(N, M):
    C = Context()
    r = C.iterates(N).repartition(M)
    ans = r.getNumPartitions()
    if C.rank == 0: assert ans == M
    pl = r.foreachPartition(len)
    assert isinstance(pl, RDD)
    u = pl.collect()
    if C.rank == 0:
        u = np.array(u)
        assert np.all(u >= N//M)
        assert u.sum() == N

def test_byKey(N, M):
    C = Context()
    # sort by keys themselves
    r = C.iterates(N).byKey(lambda x: x//5, M)
    assert isinstance(r, RDD)

    ans = r.getNumPartitions()
    if C.rank == 0: assert ans == M
    pl = r.foreachPartition( lambda p: print(p.seq, [x for x in p]) )

def main():
    test_iterate(100)
    test_parquet()
    test_repartition(99, 2)
    test_repartition(100, 20)
    test_repartition(101, 10)
    test_repartition(3, 7)
    test_repartition(72, 128)

    test_byKey(100, 20)

if __name__ == "__main__":
    main()
