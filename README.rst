MPI-RDD
=======

This project is intended to provide an RDD-like abstraction
for working with datasets distributed over MPI ranks.

Example Code::

    from rdd import *

    def store(p):
        fmt = "out%d.pq"
        pd.DataFrame(p, columns = ["name", "value"]).to_parquet(fmt % p.seq)

    C = Context() # Create a context (holding MPI comm and rank information)
                  # In particular, it has C.rank and C.procs as myrank and #ranks
    

    r = C . iterates(3000)                 # numbers 0, ..., 2999 round-robin across all ranks
          . map(lambda i: f"input.{i}.pq") # turn numbers into strings (filenames here)
          . partitionFrom(readParquet)     # use readParquet to create 1 partition per item
          . repartition(1000)              # move data to create exactly 1000 output partitions
          . foreachPartition(store)        # run "store" on every partition

Some explanations: C.iterates() is an RDD class, and
each method call after C.iterates() creates another RDD class.
Therefore the above code creates 5 distinct RDDs.

Every RDD contains a list of Partition classes.
The only data each MPI rank stores locally are its own
Partitions.  Most functions work on local partitions only,
and thus require no communication.  Map and filter
are trivially parallel, for example.

The above code only does MPI communication for the
"repartition" function call.  That call will move data
between ranks so as to achieve 1000 partitions.
It currently exactly balances the partitions, so that
each partition has an equal number of rows (+/- 1 for non-divisible
situations).

For RDD-s before `repartition`, each rank has `3000 / procs` partitions.
For RDDs after the repartition call, each rank has `1000 / procs` partitions.
