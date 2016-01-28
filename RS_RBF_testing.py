"""
RBF testing
"""
if __name__ == '__main__':
    
    import numpy as np
    from RS_RBF import RBF
    
    ### some testing things to see it works
    print("Testing for RBF: ")
    tops = {'method':'scale01',
            'box':[[0,0],[2,2]]} # shift/scale points to [0,2]^2
                                 # see transform.py for more info
    test = RBF('linear',1,1)#,transform_options=tops)
    samples = np.array( [[1,2],[3,4]])
    vals = np.array( [[1], [2]])
    test.addSamples(samples,vals)
    moresamples = np.array([[10,8.2]])
    morevals = np.array([[5.61]])
    test.addSamples(moresamples,morevals)
    print("Here are the current samples")
    print(test.getSamples())
    print("Here are the current values")
    print(test.getValues())
    print("Building model")
    test.buildModel()
    print("Testing model: ")
    print("value of samples[0] should be ",vals[0]," calculated as ",test.interp(samples[0]))
    print("value of samples[1] should be ",vals[1]," calculated as ",test.interp(samples[1]))
    print("value of moresamples should ",morevals," calculated as ",test.interp(moresamples))
    print("At some other random points: ")
    print("value at [2,3] is: ",test.interp(np.array([[2,3]])))
    print("value at [10,0], [0,10] are: ",test.interp(np.array([[10,0],[0,10]])))


    ### TIMING
    import matplotlib.pyplot as plt
    from time import time
    
    d = 2
    neval = 10000

    nstart = int(np.ceil(np.log2(d*3)))+3
    locs  = np.random.rand(neval,d)
    tbuild = []
    teval  = []
    nvals  = []
    for n in [i for i in range(nstart,max(nstart+5,10))]:
        print('n: ',n)
        npts = 2**n
        samples = np.random.rand(npts,d)
        values  = np.random.rand(npts,1)
        test = RBF()
        test.addSamples(samples,values)
        t1 = time()
        test.buildModel()
        t2 = time()
        test.interp(locs)
        t3 = time()
        nvals.append(n)
        tbuild.append(t2-t1)
        teval.append((t3-t2)/neval)

    plt.figure(1)
    plt.subplot(211)
    plt.title('build time')
    plt.ylabel('log2(time [s])')
    plt.plot(nvals,np.log2(tbuild),'bo-')
    plt.subplot(212)
    plt.title('average evaluation time')
    plt.ylabel('log2(time [ms])')
    plt.xlabel('log_2(samples)')
    plt.plot(nvals,np.log2([t*1000 for t in teval]),'rx-')
    plt.show()
