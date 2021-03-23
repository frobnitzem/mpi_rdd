# Implement nearest fill rule

import numpy as np

# Return the fan-in / fan-out communication strategy for achieving delta == 0
# 
# \param  delta = items - target
# \return [ [(src,dst,count)] ] = sending schedule at each level
def fill(delta):
    N = len(delta)
    assert np.sum(delta) == 0

    lev = [ delta ] # surplus items at each level
    sends = []

    level = 0 # target level for sends
    skip = 1  # skip for addressing within target level
    while N > 1:
        sno = []
        nlev = []

        odd = len(lev[-1]) % 2
        for i in range(0,len(lev[-1])-odd, 2):
            c0 = lev[-1][i]
            c1 = lev[-1][i+1]
            if c1 > 0: # surplus items
                sno.append( ((i+1)*skip,i*skip,c1) )
            nlev.append( c0+c1 )

        if odd: # copy-through
            nlev.append(lev[-1][-1])

        if len(sno) > 0:
            sends.append(sno)
        lev.append(nlev) # surplus/deficit at ea. level
        N = (N+1)//2
        skip *= 2
        level += 1
        assert N == len(nlev)
        assert level+1 == len(lev)

    assert len(lev[-1]) == 1 and lev[-1][0] == 0

    # N = 1
    while level > 0:
        N *= 2
        skip = skip//2 # skip for addressing within target level
        level -= 1 # target level for sends
        sno = []

        odd = len(lev[level]) % 2
        for i in range(0, len(lev[level])-odd, 2):
            c0 = lev[level][i]
            c1 = lev[level][i+1]
            if c1 < 0:
                sno.append( (i*skip, (i+1)*skip, -c1) )

        if len(sno) > 0:
            sends.append(sno)

    return sends

def check_fill(delta, sends):
    x = delta.copy()
    # follow each send
    for sno in sends:
        for i,j,n in sno:
            assert x[i] >= n, "Sending non-existent items"
            x[i] -= n
            x[j] += n
    assert np.all(x == 0), "Improper ending state"

def test_fill():
    ans = fill(np.array([0])) # test trivial case
    print(ans)

    # test a known case
    delta = np.array([-2,3,-1,1, 1,-5,3])
    sends = fill(delta)
    print(delta)
    print(sends)
    check_fill(delta, sends)

    for M in range(2,100,3): # range of sizes
        for j in range(10): # tests per size
            delta = np.random.randint(-10,11, size=M)
            v = np.random.randint(M)
            delta[v] -= delta.sum() # deposit excess

            sends = fill( delta )
            check_fill(delta, sends)

if __name__=="__main__":
    test_fill()

