********************************************************************************

Copyright (C) 2014 Cornell University

Response_Surface is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/gpl-2.0.html>

********************************************************************************

Contact:
Kyle Perline, krp73@cornell.edu

********************************************************************************


This is a general API for response (or regression) surfaces.  Suppose we have a set of n samples in d-dimensions represented as an nXd matrix S and a set of n corresponding values in v-dimensions represented as an nXv matrix V.  A response surface is a 'best-fit' f:R^d -> R^v based on (S,V). 

There are two main components:

1. Transformation of S and V to S' and V'.  For example, some response surface methods perform better if S and V are first normalized to the unit hypercubes.

2. Response surface best-fits.  For example, polynomial regression or Radial Basis Function regression.

For an introduction of how to use this, see RS_poly_testing.py.

For details on how to construct a new transformation function, see transformation.py.  
For an example of how to construct a new response surface method, see both RS_Parent.py and RS_poly.py.