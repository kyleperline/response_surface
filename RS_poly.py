import numpy as np
from RS_Parent import RS_Parent
from general_functions import *

class poly(RS_Parent):

    def __init__(self,deg=1,**kwargs):
        super().__init__(deg,**kwargs)
        self.mops['model requires validation set'] = False

    def get_default_mops(self,*args,**kwargs):
        degree = args[0]
        mops = {'deg':degree}
        return mops

    def get_default_tops(self,*args,**kwargs):
        tops ={'method':'scale01_domain_range',
               'tranform_auto':True}
        return tops

    def make_model(self):
        """
        We have the system y=Xb (X comes from samples, y=values)
        Solve b-hat = (X'X)^-1 X' y
        do this with linalg.lstsq
        """
        # get X, the polynomial tail
        X = make_poly(self.samples, self.mops['deg'])
        # compute and store b:
        self.model = np.linalg.lstsq(X, self.values)[0]
        return {}

    def perform_interp(self, locs):
        X = make_poly(make2d(locs),self.mops['deg'])   
        return make2d(np.dot(X,self.model))


