if __name__=='__main__':
    
    import numpy as np
    from RS_poly import poly
    
    def doubleNeg(array):
        return np.hstack((array,-array))
    ### do some testing/demonstrations
    # set numpy print options so it's easier to see the results
    np.set_printoptions(linewidth=220)
    # make 1-d samples and values
    samples1 = np.reshape(np.linspace(0,10,3),(-1,1))
    samples2 = np.reshape(np.linspace(11,18,4),(-1,1))
    value1 = doubleNeg(samples1**2)
    value2 = doubleNeg(samples2**3)
    locs   = np.reshape(np.linspace(0,10,5),(-1,1))
    tops = {'method':'scale01',
            'box01':True}
    
    I = poly(1,transform_options=tops)
    print('We will fit a function from R^1 -> R^2 with polynomial regression')
    print('PART 1: deg=1, transform=map to [0,1], v=k**2')
    print('adding first set of samples, values, where v=k**2')
    I.addSamples(samples1,value1)
    print('building the model using poly_order=1')    
    I.buildModel()
    print('here are the original samples and values (x,y1,y2)')
    X = I.getSamples()
    Y = I.getValues()
    print(np.hstack((X,Y)))
    print('here are the transformed samples and values (x,y1,y2)')
    X = I.getSamplesTransformed()
    Y = I.getValuesTransformed()
    print(np.hstack((X,Y)))
    print('observe how the transformed samples are on [0,1] and the values remain the same')
    print('interpolating locs: \n',locs.T)  
    v = I.interp(locs)
    print('received these (x,y1,y2) (again, with order 1 fit): ')
    print(np.hstack((locs,v)))
    
    print('\nPART 2: make poly_order=2')
    print('now building the model again, this time with order 2 fit')
    I.mops['deg']=2
    I.buildModel()
    print('and now performing the same interpolation again')
    v = I.interp(locs)
    print(np.hstack((locs,v)))
    print('observe how the interpolated values are now quadratic')

    print('\nPART 3: add more samples that are v=k**3')
    print('adding another set of samples')
    I.addSamples(samples2,value2)
    print('building the model again, still with order 2 fit')
    I.buildModel()
    print('here are the original samples and values')
    X = I.getSamples()
    Y = I.getValues()
    print(np.hstack((X,Y)))
    print('here are the transformed samples and values')
    X = I.getSamplesTransformed()
    Y = I.getValuesTransformed()
    print(np.hstack((X,Y)))
    print('observe how the transformation was recomputed to map the samples onto [0,1]')

    print('\nPART 4: switch to transform=identity map')
    print('now we will make the transformation the identity map and rebuild')
    I.tops['method']='ID'
    I.buildModel()
    print('here are the original samples and values')
    X = I.getSamples()
    Y = I.getValues()
    print(np.hstack((X,Y)))
    print('here are the transformed samples and values')
    X = I.getSamplesTransformed()
    Y = I.getValuesTransformed()
    print(np.hstack((X,Y)))
    print('observe that the transformed samples are the same as the original samples as desired, because we now use the identity map')
    print('now, we perform the same interpolations as before')
    print('note that the new samples,values were v=k**3, which changes the model')
    v = I.interp(locs)
    print(np.hstack((locs,v)))
