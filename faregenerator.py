import random
import numpy
import math
import fare
import copy


# base class for a fare parameter generator by default returns parameters matching the 'classic' RoboUber 
class FareParamGenerator:

      def __init__(self, world, origin, destination=None, **distparams):
          self._origin = origin
          self._destination = destination
          self._world = world

          # general setup for whatever parameters are needed to generate max cost, max wait, and destination
          self._costparams = distparams['costparams']
          self._waitparams = distparams['waitparams']
          self._destparams = distparams['destparams']

          # the generators proper. 
          self._costGen = numpy.random.Generator(numpy.random.PCG64())
          self._waitGen = numpy.random.Generator(numpy.random.PCG64())
          self._destGen = numpy.random.Generator(numpy.random.PCG64())

      @property
      def fareType(self):
          return 'default'
          

      def getDestination(self, **destparams):
          if self._destination is None:
             self._destination = self._world.nodes[round(self._destGen.uniform(low=0, high=self._world.size-1))]
          return self._destination

      def getMaxWait(self, distance_based=True, distance_weight=10, **waitparams):
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                distance_weight = 10
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 200
          return distance_term+5*self._waitGen.gamma(shape=2.0, scale=1.0)
          

      def getMaxCost(self, distance_based=True, per_minute=True, distance_weight=10, **costparams):
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if per_minute:
                return distance_weight*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             return 10*(math.dist(self._world.extent[1], self._world.extent[0]))

      def resetDestination(self):
          self._destination = None
          return

class NormalParamGenerator(FareParamGenerator):

      def __init__(self, world, origin, destination=None, **distparams):

          super().__init__(world=world, origin=origin, destination=destination, **distparams)
          ''' the **distparams dictionary should (or can) contain:
              1) a, b, and scale parameters for a beta distribution on cost (['costparams']), typically per-segment
              2) mu and sigma parameters for a Gaussian distribution on maximum wait time ['waitparams']
             3) a dict of num: (position, sigma) parameters for a product-of-2D-Gaussians distribution on destination ['destparams']  
          '''
          self._valid_dests = None

      @property
      def fareType(self):
          return 'normal'

      '''
         getDestination for a Normal fare type should give a destination that is a product-of-Gaussians
         distance-dependent function from the origin point. Generating a product of Gaussians is actually
         quite simple, because that will simply yield another Gaussian with a different mean and variance.
         But this simple step is made much more complicated by translating to a graph, because only the
         actual nodes in the graph are valid. We definitely don't want to test each individual node for
         distance from a point sampled from the resultant Gaussian, so what to do? One answer is below. 
      '''
      def getDestination(self, **destparams):
          # we *can* override the destination parameters if desired (though this is not usual)
          if destparams is not None and len(destparams) > 0:
             self._destparams = copy.deepcopy(destparams) # avoid side effects on a dictionary input
          # if the destination has already been generated, we don't need to do this again.
          elif self._destination is not None:
               return self._destination
          # default in the event of no destination parameters to a single Gaussian centred on the middle of the map
          if self._destparams is None or len(self._destparams) == 0:
             self._destparams = {'d0': (numpy.array([24, 24]), numpy.array([[12, 0], [0, 12]]))}
          # grab all of the parameters for each Gaussian term, then accumulate into a single Gaussian
          # as seen below.
          mu, sigma = self._destparams.pop('d0')   # first vector of coordinates [x, y] and covariance matrix (diagonal)
          for target in self._destparams.values(): # accumulate Gaussians
              H = numpy.linalg.inv(numpy.linalg.cholesky(sigma+target[1])) # decompose the sum of covariances into a triangular matrix for efficiency and use its inverse to get new mu, sigma
              M_1 = H@mu             # intermediate terms used in the subsequent computation.
              M_2 = H@target[0]      # Similar ideas come up in the Kalman Filter for those interested.
              S_1 = H@sigma
              S_2 = H@target[1]
              S_1T = numpy.transpose(S_1) # was matrix_transpose but this function not supported on Numpy < 2.0          
              mu = numpy.transpose(S_2)@M_1 + S_1T@M_2 # finally, assemble the new mean and covariance from the intermediate terms
              sigma = S_1T@S_2
          self._destparams = {'d0': (mu, sigma)} # overwrite the old _destparams tuple with a single global Gaussian
          # sample from this single Gaussian to get a notional target. The initial sample will be a floating-point coordinate
          # pair, so round to the next integer. We also use a clipped Gaussian since the world doesn't extend in all
          # directions to infinity.
          # need a .astype(int) because numpy does not implement the dtype=int functionality for the rint function,
          # but (deceptively) rint itself does NOT convert to an int but keeps the value as a float.
          target = numpy.rint(numpy.clip(self._destGen.multivariate_normal(mean=self._destparams['d0'][0], cov=self._destparams['d0'][1], method='cholesky'),
                                         a_min=self._world.extent[0],
                                         a_max=self._world.extent[1])).astype(int)
          return self.getValidGraphPoint(target)


          
      ''' we need to get a real point on the graph. The algorithm to do so is as follows: 
          we find the next row/column in both N-S and E-W destinations from the target point, that has
          any nodes in the graph. Then we find the nearest point in the orthogonal direction (depending
          on whether we selected a row or column). We are guaranteed that this point will be the closest.
          We also get the index for free, since it's the (column, row) offset from the target. Also we
          need to allow for multiple closest nodes, so we will accumulate a list of candidates and sample
          randomly from them to produce a final destination.
      '''
      def getValidGraphPoint(self, target):          
          # create a bitmap matrix for the possible destinations
          if self._valid_dests is None or self._valid_dests.shape != (self._world.extent[1][0], self._world.extent[1][1]):
             # it would be very useful if Numpy had some means of constructing an array of this type, where each element's
             # value is conditional upon some property of the coordinates - the method below is quite inefficient since it
             # involves copying from a nested list. But neither numpy.fromfunction nor numpy.fromiter will do this right,
             # because they expect to evaluate an expression uniformly over the entire array.
             self._valid_dests = numpy.array([[1 if (x, y) in self._world.locations else 0 for x in range(self._world.extent[1][0]+1)] for y in range(self._world.extent[1][1]+1)], dtype=int)
          # initialise the array of candidates
          matches = []
          # then follow the algorithm above on the bitmap matrix. Using generator expressions should speed things up
          # quite a bit by taking advantage of early stopping.
          matchidx_S = next(row for row in range(target[1], self._valid_dests.shape[0]) if numpy.any(self._valid_dests[row]))
          min_dist_a = numpy.min(numpy.argwhere(self._valid_dests[matchidx_S,target[0]:]==1), initial=1000000000)
          min_dist_b = numpy.min(numpy.argwhere(self._valid_dests[matchidx_S,target[0]-1::-1]==1), initial=1000000000)+1
          # having gathered the 2 minima in a Southward direction, get the Euclidean distance to the nearer
          min_dist_S = math.sqrt(min(min_dist_a, min_dist_b)**2+(matchidx_S-target[1])**2)
          try:
             matchidx_N = next(row for row in range(target[1]-1, max(-1, target[1]-math.trunc(min_dist_S)-1), -1) if numpy.any(self._valid_dests[row]))
             min_dist_c = numpy.min(numpy.argwhere(self._valid_dests[matchidx_N,target[0]:]==1), initial=1000000000)
             min_dist_d = numpy.min(numpy.argwhere(self._valid_dests[matchidx_N,target[0]-1::-1]==1), initial=1000000000)+1
             min_dist_N = math.sqrt(min(min_dist_c, min_dist_d)**2+(target[1]-matchidx_N)**2)
          except StopIteration:
             min_dist_N = 1000000000.0 # set to something large (but not max float because will be used in computation later)
          try:
             matchidx_E = next(col for col in range(target[0], min(self._valid_dests.shape[1], target[0]+math.trunc(min_dist_N), target[0]+math.trunc(min_dist_S))) if numpy.any(self._valid_dests[:,col]))
             min_dist_f = numpy.min(numpy.argwhere(self._valid_dests[target[1]:,matchidx_E]==1), initial=1000000000)
             min_dist_g = numpy.min(numpy.argwhere(self._valid_dests[target[1]-1::-1,matchidx_E]==1), initial=1000000000)+1
             min_dist_E = math.sqrt(min(min_dist_f, min_dist_g)**2+(matchidx_E-target[0])**2)
          except StopIteration:
             min_dist_E = 1000000000.0
          try: 
             matchidx_W = next(col for col in range(target[0]-1, max(-1, target[0]-math.trunc(min_dist_N)-1, target[0]-math.trunc(min_dist_S)-1, target[0]-math.trunc(min_dist_E)-1)) if numpy.any(self._valid_dests[:,col]))
             min_dist_h = numpy.min(numpy.argwhere(self._valid_dests[target[1]:,matchidx_W]==1), initial=1000000000)
             min_dist_i = numpy.min(numpy.argwhere(self._valid_dests[target[1]-1::-1,matchidx_W]==1), initial=1000000000)+1
             min_dist_W = math.sqrt(min(min_dist_h, min_dist_i)**2+(target[0]-matchidx_W)**2)
          except StopIteration:
             min_dist_W = 1000000000.0
          # Get the absolute closest point (which could be multiple)
          min_dists = numpy.array([min_dist_S, min_dist_N, min_dist_E, min_dist_W])
          min_dist_T = numpy.flatnonzero(min_dists == min_dists.min())
          # all the possibles gathered. Now select the actual minimum distances
          if 0 in min_dist_T:
             idxs = [(target[0]+min_dist_a, matchidx_S), (target[0]-min_dist_b, matchidx_S)]
             if min_dist_a != 0 and min_dist_a == min_dist_b:
                matches.extend(idxs)
             else:
                matches.extend([idxs[0]] if min_dist_a < min_dist_b else [idxs[1]])
          if 1 in min_dist_T:
             idxs = [(target[0]+min_dist_c, matchidx_N), (target[0]-min_dist_d, matchidx_N)]
             if min_dist_c != 0 and min_dist_c == min_dist_d:
                matches.extend(idxs)
             else:
                matches.extend([idxs[0]] if min_dist_c < min_dist_d else [idxs[1]])
          if 2 in min_dist_T:
             idxs = [(matchidx_E, target[1]+min_dist_f), (matchidx_E, target[1]-min_dist_g)]
             if min_dist_f != 0 and min_dist_f == min_dist_g:
                matches.extend(idxs)
             else:
                matches.extend([idxs[0]] if min_dist_f < min_dist_g else [idxs[1]])
          if 3 in min_dist_T:
             idxs = [(matchidx_W, target[1]+min_dist_h), (matchidx_W, target[1]-min_dist_i)]
             if min_dist_h != 0 and min_dist_h == min_dist_i:
                matches.extend(idxs)
             else:
                matches.extend([idxs[0]] if min_dist_h < min_dist_i else [idxs[1]])
          if len(matches) > 0:
             self._destination = self._world.getNode(*matches[random.randrange(0, len(matches))])
             return self._destination
          # bottomed out. Somehow a value was not found, so just choose at random from the valid list.
          return super().getDestination(self)

      def getMaxWait(self, distance_based=False, distance_weight=None, **waitparams):
          if waitparams is not None and len(waitparams) > 0:
             self._waitparams = copy.deepcopy(waitparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                distance_weight = 10
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 0
          return distance_term + self._waitGen.normal(loc=self._waitparams['mu'], scale=self._waitparams['sigma'])

      def getMaxCost(self, distance_based=True, per_minute=False, distance_weight=10, **costparams):
          if costparams is not None and len(costparams) > 0:
             self._costparams = copy.deepcopy(costparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None or distance_weight == 0:
                distance_weight = 10
             if per_minute:
                return distance_weight*self._costGen.beta(a=self._costparams['a'], b=self._costparams['b'])*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._costGen.beta(a=self._costparams['a'], b=self._costparams['b'])*self._world.distance2Node(self._origin, self._destination)
          else:
             return self._costparams['scale']*self._costGen.beta(a=self._costparams['a'], b=self._costparams['b'])


 # The RichParamGenerator will inherit from the NormalParamGenerator so as to grab its method for selecting a valid graph point from a 2-D sample.
class RichParamGenerator(NormalParamGenerator):

      def __init__(self, world, origin, destination=None, **distparams):

          super().__init__(world=world, origin=origin, destination=destination, **distparams)
          ''' the **distparams dictionary should (or can) contain:
              1) a, b, and scale parameters for a beta distribution on cost (['costparams']), typically for total cost
              2) min and max parameters for a uniform distribution on maximum wait time ['waitparams']
              3) a dict of num: (position, sigma, weight) parameters for a mixture-of-2D-Gaussians distribution on destination ['destparams']              
          '''
          self._setupSamplers()

      @property
      def fareType(self):
          return 'rich'

      # for the destination, we are going to be taking a sample from a mixture of 2-D Gaussians. This is easiest to do by intialising
      # some key samplers immediately
      def _setupSamplers(self):
          weightNorm = sum([val[2] for val in self._destparams.values()])  # normalisation factor for weights (so that they will sum to 1)
          # setup for the samplers themselves. A bit of (possibly paranoid) care here: we need to be absolutely sure the samplers will
          # line up with their respective weights, but as will be seen, this needs 2 separate arrays. We can guarantee this by first
          # initialising a list with all of these values,
          sampleComponents = [(val[2]/weightNorm, val[0], val[1], numpy.random.Generator(numpy.random.PCG64())) for val in self._destparams.values()]
          # grouping the individual component samplers with their means and covariances in one array,
          self._samplers = tuple([(c[1], c[2], c[3]) for c in sampleComponents])
          # and grouping the individual weights in the second array
          self._weights = numpy.array([c[0] for c in sampleComponents])

      # getDestination will pick a valid location by:
      # 1) sampling independently from each of the separate mixture distributions in the generator
      # 2) picking a weighted sample from one of these independent samples
      # 3) passing that sample to the nearest-valid-point algorithm (inherited from NormalParamGenerator)
      def getDestination(self, **destparams):
          if destparams is not None and len(destparams) > 0:
             self._destparams = copy.deepcopy(destparams) # avoid side effects on dictionary input
             self._setupSamplers()
          elif self._destination is not None:
               return self._destination
          # Do the independent sampling. This uses the set up generators
          samples = numpy.clip(numpy.array([sample[2].multivariate_normal(mean=sample[0], cov=sample[1]) for sample in self._samplers]), a_min=self._world.extent[0], a_max=self._world.extent[1])
          # Now pick one of those values (this is why we need the weights separately, because they are used in this step)
          target = numpy.rint(self._destGen.choice(samples, p=self._weights, axis=0)).astype(int)
          return self.getValidGraphPoint(target)
              

      def getMaxWait(self, distance_based=False, distance_weight=None, **waitparams):
          if waitparams is not None and len(waitparams) > 0:
             self._waitparams = copy.deepcopy(waitparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                distance_weight = 5    # default rich weighting half as much (i.e. distance is not so much of a factor in wait time) 
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 0
          return distance_term + self._waitGen.uniform(low=self._waitparams['min'], high=self._waitparams['max'])

      def getMaxCost(self, distance_based=False, per_minute=False, distance_weight=None, **costparams):
          if costparams is not None and len(costparams) > 0:
             self._costparams = copy.deepcopy(costparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None or distance_weight == 0:
                distance_weight = 20 # default rich will pay twice as much for the same distance as normal
             if per_minute:
                return distance_weight*self._costGen.beta(a=self._costparams['a'], b=self._costparams['b'])*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._costGen.beta(a=self._costparams['a'], b=self._costparams['b'])*self._world.distance2Node(self._origin, self._destination)
          else:
             return self._costparams['scale']*self._costGen.beta(a=self._costparams['a'], b=self._costparams['b'])

class HurryParamGenerator(FareParamGenerator):

      def __init__(self, world, origin, destination=None, **distparams):

          super().__init__(world=world, origin=origin, destination=destination, **distparams)

          ''' the **distparams dictionary should (or can) contain:
              1) a ceiling parameter for total cost (['costparams'])
              2) tau parameter for an exponential distribution on maximum wait time ['waitparams']
              3) mu and sigma parameters for a Gaussian distribution offset by mu distance from source, on destination ['destparams']  
          '''

      @property
      def fareType(self):
          return 'hurry'

      def getDestination(self, **destparams):
          if destparams is not None and len(destparams) > 0:
             self._destparams = copy.deepcopy(destparams) # avoid side effects on dictionary input
          elif self._destination is not None:
               return self._destination
          r_ref = self._destGen.normal(loc=self._destparams['mu'], scale=self._destparams['sigma'])
          if r_ref < 0.71:
             r_ref = 0.71   # don't select the origin as the destination. The 0.71 factor arises because
                            # if a node is connected only by diagonals, halfway to its nearest neighbour
                            # will be 1/sqrt(2) away (for convenience we choose a fixed constant just slightly
                            # larger than 1/sqrt(2)).
          '''
              what we want to do next is to take a random selection from those nodes in self._world.nodes
              that are closest to r_ref distance away from the origin. So:
                 Get all values where:
                     min(sqrt((node.index[0]-origin.index[0])**2 + (node.index[1]-origin.index[1])**2)-r_ref) is True
                 Choose a node from this reduced subset of self._world.nodes. 
          '''
          ''' esoteric: the next 3 lines compute the array of distances. einsum is a quirky but powerful
              numpy function that computes generic sums of products, where the notation is specified as
              the first argument and reads like this: any pairwise index sequence such as ii or ij multiplies
              the array elements together, comma-separated values such as i, j add the elements together, 
              and the explicit indicator such as ->j tells you which outputs of the generic matrix so
              created are output by picking out rows or columns or whatever according to the indices
              listed after the ->. So a blank -> will collapse the output to a scalar (e.g. dot product),
              ->j will pull each row, etc.
              See: https://numpy.org/doc/2.3/reference/generated/numpy.einsum.html#numpy.einsum
              and also https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
              for the rationale behind this
          '''
          destArray = numpy.array(self._world.locations)
          xydists = destArray-numpy.array(self._origin.index)
          rdists = numpy.sqrt(numpy.einsum('ij,ij->j', xydists, xydists))
          ''' more obscurity: now we get an array of distance deltas for each computed
              distance, relative to r_ref, and extract the indices of the resultant
              array that correspond to the minval. Then we take a random choice of these
              indices and use it to index the main nodes array (whose array indices line
              up with rdists) in order to get a random destination. There will be a prize 
              for anyone who can optimise this and prove that it is the optimal approach!
          '''
          rdists = numpy.fabs(rdists-r_ref)
          minvals = numpy.argwhere(rdists == numpy.min(rdists))
          self._destination = self._world.nodes[self._destGen.choice(minvals)[0]]
          return self._destination

      def getMaxWait(self, distance_based=False, distance_weight=None, **waitparams):
          if waitparams is not None and len(waitparams) > 0:
             self._waitparams = copy.deepcopy(waitparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                distance_weight = 1    # default hurry weighting very small (1/10 normal, distance barely matters in wait time) 
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 0
          return distance_term + self._waitGen.exponential(scale=self._waitparams['tau'])

      def getMaxCost(self, distance_based=False, per_minute=False, distance_weight=None, **costparams):
          if costparams is not None and len(costparams) > 0:
             self._costparams = copy.deepcopy(costparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None or distance_weight == 0:
                distance_weight = 20 # default hurry will pay twice as much for the same distance as normal
             if per_minute:
                return distance_weight*self._costparams['ceiling']*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._costparams['ceiling']*self._world.distance2Node(self._origin, self._destination)
          else:
             return self._costparams['ceiling']

class BudgetParamGenerator(FareParamGenerator):

      def __init__(self, world, origin, destination=None, **distparams):

          super().__init__(world=world, origin=origin, destination=destination, **distparams)

          ''' the **distparams dictionary should (or can) contain:
              1) a ceiling parameter for maximum cost (['costparams']), typically per segment
              2) mu and sigma parameters for a Gaussian distribution on maximum wait time ['waitparams'], typically distance-adjusted
              3) mu and sigma parameters for a Gaussian distribution offset by mu distance from source, on destination ['destparams']  
          '''

      @property
      def fareType(self):
          return 'budget'
         
      def getDestination(self, **destparams):
          if destparams is not None and len(destparams) > 0:
             self._destparams = copy.deepcopy(destparams) # avoid side effects on a dictionary input
          elif self._destination is not None:
               return self._destination
          ''' the destination distribution for the Budget fare type has the same form as for the Hurry
              fare type, the only difference being a somewhat wider normal distribution with a typically
              longer-distance mean (reflecting the fact that Budget users consider the value to be 
              principally in relatively long-distance travel). All of this is only evident in the 
              _destparams dictionary.
          '''
          r_ref = round(self._destGen.normal(loc=self._destparams['mu'], scale=self._destparams['sigma']))
          if r_ref < 1:
             r_ref = 1   # don't select the origin as the destination
          destArray = numpy.array(self._world.locations)
          xydists = destArray-self._origin.index
          rdists = numpy.sqrt(numpy.einsum('ij,ij->j', xydists, xydists))
          rdists = numpy.fabs(rdists-r_ref)
          minvals = numpy.argwhere(rdists == numpy.min(rdists))
          self._destination = self._world.nodes[self._destGen.choice(minvals)[0]]
          return self._destination

      def getMaxWait(self, distance_based=True, distance_weight=10, **waitparams):
          if waitparams is not None and len(waitparams) > 0:
             self._waitparams = copy.deepcopy(waitparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                distance_weight = 10    # default budget distance weight is normal 
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 0
          return distance_term + self._waitGen.normal(loc=self._waitparams['mu'], scale=self._waitparams['sigma'])

      def getMaxCost(self, distance_based=True, per_minute=False, distance_weight=10, **costparams):
          if costparams is not None and len(costparams) > 0:
             self._costparams = copy.deepcopy(costparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None or distance_weight == 0:
                distance_weight = 10 # default budget scales distances into cost as normal
             if per_minute:
                return distance_weight*self._costparams['ceiling']*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._costparams['ceiling']*self._world.distance2Node(self._origin, self._destination)
          else:
             return self._costparams['ceiling']


class OpportuneParamGenerator(NormalParamGenerator):

      def __init__(self, world, origin, destination=None, **distparams):

          super().__init__(world=world, origin=origin, destination=destination, **distparams)

          ''' the **distparams dictionary should (or can) contain:
              1) min and max parameters for a uniform distribution on cost (['costparams']), typically per segment
              2) a tau parameter for an exponential distribution on maximum wait time ['waitparams']
              3) a tuple of (position, sigma) parameters for a product-of-Gaussians distribution on destination (assumed zero-mean on position) ['destparams']

              the Opportune fare class uses the product-of-Gaussians random destination generator from the Normal class (but with tighter typical sigma terms)   
          '''

      @property
      def fareType(self):
          return 'opportune'

      def getMaxWait(self, distance_based=False, distance_weight=None, **waitparams):
          if waitparams is not None and len(waitparams) > 0:
             self._waitparams = copy.deepcopy(waitparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                distance_weight = 5  # default opportune distance weight is half of normal; opportune Fares are more indifferent to distance  
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 0
          return distance_term + self._waitGen.exponential(scale=self._waitparams['tau'])

      def getMaxCost(self, distance_based=True, per_minute=True, distance_weight=20, **costparams):
          if costparams is not None and len(costparams) > 0:
             self._costparams = copy.deepcopy(costparams)
          if distance_based:
             if distance_weight is None or distance_weight == 0:
                distance_weight = 20 # default opportune scales distances into cost at twice normal - paying more for convenience
             if self._destination is None:
                self.getDestination()
             if per_minute:
                return distance_weight*self._costGen.uniform(low=self._costparams['min'], high=self._costparams['max'])*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._costGen.uniform(low=self._costparams['min'], high=self._costparams['max'])*self._world.distance2Node(self._origin, self._destination)
          else:
             return self._costGen.uniform(low=self._costparams['min'], high=self._costparams['max'])

class RandomParamGenerator(FareParamGenerator):

      def __init__(self, world, origin, destination=None, **distparams):

          super().__init__(world=world, origin=origin, destination=destination, **distparams)

          ''' the **distparams dictionary should (or can) contain:
              1) min and max parameters for a uniform distribution on cost (['costparams'])
              2) min and max parameters for a uniform distribution on maximum wait time (['waitparams'])

              the Random fare class uses the default uniform random destination generator from the parent class.    
          '''

      @property
      def fareType(self):
          return 'random'

      def getMaxWait(self, distance_based=False, distance_weight=None, **waitparams):
          if waitparams is not None and len(waitparams) > 0:
             self._waitparams = copy.deepcopy(waitparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None:
                if 'minwaitd' not in self._waitparams:
                   self._waitparams['minwaitd'] = 1
                if 'maxwaitd' not in self._waitparams:
                   self._waitparams['maxwaitd'] = 20
                distance_weight = self._waitGen.uniform(low=self._waitparams['minwaitd'], high=self._waitparams['maxwaitd'])  # default random distance weight is ... random.  
             distance_term = distance_weight*self._world.distance2Node(self._origin, self._destination)
          else:
             distance_term = 0
          return distance_term + self._waitGen.uniform(low=self._waitparams['min'], high=self._waitparams['max'])

      def getMaxCost(self, distance_based=False, per_minute=False, distance_weight=None, **costparams):
          if costparams is not None and len(costparams) > 0:
             self._costparams = copy.deepcopy(costparams)
          if distance_based:
             if self._destination is None:
                self.getDestination()
             if distance_weight is None or distance_weight == 0:
                if 'mincostd' not in self._waitparams:
                   self._waitparams['mincostd'] = 5
                if 'maxcostd' not in self._waitparams:
                   self._waitparams['maxcostd'] = 20
                distance_weight = self._costGen.uniform(low=self._waitparams['mincostd'], high=self._waitparams['maxcostd'])  # default random distance weight is ... random. 
             if per_minute:
                return distance_weight*self._costGen.uniform(low=self._costparams['min'], high=self._costparams['max'])*self._world.travelTime(self._origin, self._destination)
             else:
                return distance_weight*self._costGen.uniform(low=self._costparams['min'], high=self._costparams['max'])*self._world.distance2Node(self._origin, self._destination)
          else:
             return self._costGen.uniform(low=self._costparams['min'], high=self._costparams['max'])

class FlatParamGenerator(FareParamGenerator):

      def __init__(world, origin, destination=None, **distparams): #wait=None, cost=None):

          super().__init__(self, world=world, origin=origin, destination=destination, **distparams)
          ''' the **distparams dictionary should (or can) contain:
              1) a fixed upper cost (['costparams'])
              2) a flat wait time (['waitparams'])
          '''

      @property
      def fareType(self):
          return 'flat'

      # a flat Fare type will take a uniform random distribution on location - the default for the parent class.

      # default flat max wait is 2 times the longest hypothetically possible path     
      def getMaxWait(self):
          if self._waitparams is None or len(self._waitparams) == 0:
             fixedOrigin = self._world.extent[0]
             fixedDest = self._world.extent[1]
             self._waitparams['wait'] = 2*(fixedDest[0]-fixedOrigin[0]+fixedDest[1]-fixedOrigin[1])
          return self._waitparams['wait']

      # default flat max cost is the global default assuming no distance-based or time-based dependencies
      def getMaxCost(self):
          if self._costparams is None or len(self._costparams) == 0:
             self._costparams['cost'] = super().getMaxCost(distance_based=False, per_minute=False)
          return self._costparams['cost']

          
class FareGenerator:

      # fare types is a dict containing items of this form: {type label: {'type_prob': type_likelihood, 'destparams' :distance_params, 'waitparams': wait_params, 'costparams': cost_params}}
      # def __init__(self, parent_node, base_prob, fare_types=(("normal", 1.0, ("Gaussian", 0, 10), ("Gaussian", 50, 20), ("beta", 1.0, 1.0, "per_segment")),)):
      def __init__(self, world, parent_node, base_prob, fare_file=None, **fare_types):

          self._world = world
          self._parent = parent_node
          self._baseProb = base_prob
          self._baseGen = numpy.random.default_rng()
          self._typeGen = numpy.random.default_rng()
          self._fileOut = fare_file
          tempIdxVals = []
          tempFareTypes = []
          cumulativeProb = 0.0
          # No fare types specified => use the default generator (only).
          if fare_types is None or len(fare_types) == 0:
              cumulativeProb = 1.0
              defaultParams = {'destparams': None, 'waitparams': None, 'costparams': None}
              tempIdxVals.append(1.0)
              tempFareTypes.append(FareParamGenerator(world=self._world, origin=self._parent, **defaultParams))
          else:
             # we are going to be modifying a possibly shared dictionary, so copy it first to avoid side effects
             fareTypesInt = copy.deepcopy(fare_types)
             # now find all the fare types and unpack them
             for ftype in fareTypesInt.items():
                 cumulativeProb = cumulativeProb + ftype[1].pop('type_prob')
                 tempIdxVals.append(cumulativeProb)
                 if ftype[0] == 'normal':
                    tempFareTypes.append(NormalParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 elif ftype[0] == 'rich':
                    tempFareTypes.append(RichParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 elif ftype[0] == 'hurry':
                    tempFareTypes.append(HurryParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 elif ftype[0] == 'budget':
                    tempFareTypes.append(BudgetParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 elif ftype[0] == 'opportune':
                    tempFareTypes.append(OpportuneParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 elif ftype[0] == 'random':
                    tempFareTypes.append(RandomParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 elif ftype[0] == 'flat':
                    tempFareTypes.append(FlatParamGenerator(world=self._world, origin=self._parent, **ftype[1]))
                 else:
                    raise ValueError("Invalid fare type: {0}".format(ftype[0]))
          # renormalise likelihoods to get a true probability
          tempIdxVals = numpy.array(tempIdxVals)
          if cumulativeProb != 1.0:
             tempIdxVals = tempIdxVals/cumulativeProb
          ''' the _fareTypes structure will do all the work. To get a new fare, we will simply select a random number
              on the interval [0, 1), go to that cumulative distribution point in the _fareTypes dictionary, then
              sample from the distributions given by the associated parameter generator.
          '''
          self._fareTypes = dict(zip(tempIdxVals, tempFareTypes))
          self._fareKeys = numpy.sort(tempIdxVals) # create a fast-lookup key array

      '''
          As noted, generating a Fare is very simple once the preliminaries are set up: first, sample from the base random generator
          to see if a Fare is going to be generated at all, then if it is, sample from the type random generator to get the type to
          be selected, and then finally, sample from the type's underlying parameter generator (where the bulk of the complexity lies)
          to set up the individual values for this new Fare.  
      '''
      def generateFare(self):

          if self._baseGen.uniform() < self._baseProb:
             typeSample = self._typeGen.uniform()
             # slight subtlety here: the _fareTypes dictionary is keyed by the cumulative probability of all fare types
             # up to the one indexed. This makes it possible to sample by grabbing the first key higher than some random
             # number generated by a uniform generator. 
             typeGenerator = self._fareTypes[next(T for T in self._fareKeys if T > typeSample)]
             typeGenerator.resetDestination()
             newDest = typeGenerator.getDestination()
             newWait = typeGenerator.getMaxWait()
             newCost = typeGenerator.getMaxCost()
             newFare = fare.Fare(parent=self._world,
                                 origin=self._parent,
                                 destination=newDest,
                                 call_time=self._world.simTime,
                                 wait_time=newWait,
                                 cost_ceiling=newCost)
             if self._fileOut is not None:
                print('"{0}"'.format(typeGenerator.fareType),
                      newFare.origin[0], newFare.origin[1],
                      newFare.destination[0], newFare.destination[1],
                      newWait, newCost, sep=',', file=self._fileOut, flush=True)
             return newFare
          return None

            
 
