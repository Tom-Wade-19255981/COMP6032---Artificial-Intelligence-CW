import math
import numpy
import heapq
import inspect
import threading # BUGFIX 21 October 2025 ADR needed to support stable output

from node import Node
from fare import Fare

# some straightforward data containers to help in initialising Worlds. A junction will
# end up being a Node, a street will end up being an Edge.

# a junctionDef contains:
# 1) its (x,y) coordinates
# 2) how many taxis can occupy the junction
# 3) whether a taxi is permitted to stop here
# 4) The probability of a Fare appearing
# 5) how much traffic the Node can hold before locking
# 6) traffic that 'appears' in the node (usually, at edges of the graph)
# 7) traffic that 'disappears' from the node (usually, at the edges of the graph)
class junctionDef:

      def __init__(self, x, y, cap, canStop, fareProb=None, maxTraffic=0, src=0, sink=0, **fareArgs):
          self.x = x
          self.y = y
          self.capacity = cap
          self.canStop = canStop
          self.fareProb = fareProb
          self.maxTraffic = maxTraffic
          self.tSrc = src
          self.tSink = sink
          self.fareArgs = fareArgs

# a streetDef contains:
# 1) the ID of the 2 end nodes
# 2) the expected 'outward' connection direction for both (where 'outward' describes
# the direction that would be relevant if leaving the node via that edge were possible),
# 3) whether the edge is bidirectional.
class streetDef:
      def __init__(self, nodeAIdx, nodeBIdx, dirA, dirB, biDirectional=True):
          self.nodeA = nodeAIdx # first point (origin if one-way)
          self.nodeB = nodeBIdx # second pont (destination if one-way
          self.dirA = dirA      # exit point from nodeA
          self.dirB = dirB      # exit point from nodeB
          self.bidirectional = biDirectional # one-way or 2-way street?

'''
NetWorld is the main class responsible for driving the simulation. It contains the road network
graph, the time-stepper, and the controller that deals with taxi and dispatcher commands. Notionally,
a time step is a minute of 'real time', but it could, in principle, be almost anything. The 
graph is a map of Nodes, each of which lies at some (x,y) coordinate.
'''
    
class NetWorld:

      '''
      Constructor. The x, y size of the world must be specified, all other parameters
      are optional:
      runtime - number of  time steps in the simulation (0 = run forever)
      fareprob - A default fare probability generator can be specified, this should
      be a Python dict with one mandatory key-value: 'fare_probability', plus any 
      desired arguments that will go into the **fare_types in the fare generator.
      If None is specified, then the base default generator will be used.
      jctNodes - a list of junctionDefs specifying the nodes of the graph
      edges = a list of streetDefs specifying the edges of the graph
      interpolateNodes - if True, the constructor will infer default nodes between each
      junction point for all x,y coordinates in the path between them; this will allow taxis
      to stop at any point on a street and fares to appear there. If not, only the nodes themselves
      will be generated and this means taxis can only stop at junctions and fares will
      only ever appear there.
     '''
      def __init__(self,x,y,runtime = 0, fareprob=None, jctNodes=None,edges=None,interpolateNodes=False, farefile=None):

          # size of the virtual grid. Nodes must be at (x,y) positions within the grid.
          self.xSize = x
          self.ySize = y
          # number of time steps to run. 0 means run forever.
          self.runTime = runtime
          # default fare generator is used for interpolated positions. A defined Node can
          # have its own fare probability structure
          if fareprob is None:
             fareprob = {'fare_probability': None}
          self.defaultFareGen = fareprob
          # if we are recording fares for ML data, this file will be non-None.
          self._fareFile = farefile
          # the network itself (which starts blank) is a dictionary indexed by node number
          # (a straightforward (x,y) hash) containing the (reachable) Nodes in the system
          self._net = {}
          # the traffic queue is a dictionary of entries for each node into which traffic is
          # to be injected
          self._trafficQ = {}
          # the taxi queue is a dictionary of entries for each taxi giving (node, direction)
          # ground-truth locations and admission token pairs. It looks like this therefore:
          #{taxi_obj: ((here_loc, here_dir), (admit_loc, admit_dir))
          self._taxis = {}
          # this is a dict indexed by origin of the active fares waiting for collection
          self._fareQ = {}
          # the dispatcher (there can only be one) handles allocation of fares to taxis
          self._dispatcher = None
          if jctNodes is not None:
             self.addNodes(jctNodes)
          if edges is not None:
             self.addEdges(edges,interpolateNodes)
          #self.eventQ = NetEventQueue()
          # the simulation clock. 
          self._time = 0

      # properties

      # simTime gives the current time tick
      @property
      def simTime(self):
          return self._time

      # size of the world in number of nodes.
      @property
      def size(self):
          return len(self._net)

      # extrema of the world. These may NOT be the extreme reachable points!
      @property
      def extent(self):
          try:
              return self._extent
          except AttributeError:
              xvals = [x[0] for x in self._net.keys()]
              yvals = [y[1] for y in self._net.keys()]
              self._extent = ((min(xvals), min(yvals)), (max(xvals), max(yvals)))
              return self._extent

      @property
      def locations(self):
          try:
              return self._locations
          except AttributeError:
              self._locations = tuple(self._net.keys())
              return self._locations

       # all reachable nodes in the system. Note because a dict's values() outputs an *iterator*, not an iterable,
       # before returning it, it is turned into an (immutable) iterable tuple
      @property
      def nodes(self):
          # we expect this to be called multiple times, so it makes sense to build the node list once only.
          # but this can't be done at construction time because the network is built iteratively later.
          # So instead we use the try-to-access, store-if-doesn't already-exist approach.
          try:
              return self._nodes
          except AttributeError:
             self._nodes = tuple(self._net.values())
             return self._nodes

      # this world's dispatcher    
      @property
      def dispatcher(self):
          return self._dispatcher

      # only get the taxi objects, not the admission information
      @property
      def taxis(self):
          return self._taxis.keys()

      #__________________________________________________________________________________________________________
      # methods to build the graph and place agents in it

      # takes a list of junctionDefs and adds it to the network as Nodes. Any nodes will overwrite
      # existing Nodes at the same location. This does not affect incoming links, but it will clear
      # affect outgoing ones, so the expectation is that addNodes will be followed by an addEdges
      # that reinstates any outgoing edges from this (junction) node.
      def addNodes(self, nodes):

          # validate node list before trying to set
          if isinstance(nodes, list) or isinstance(nodes, tuple):
             try:
                 invalid = next(node for node in nodes if not isinstance(node, junctionDef))
                 print("Invalid node list to add to NetWorld graph: at least one object is not a junctionDef")
                 return
             # passed validation: we have a list of nodes.
             except StopIteration:       
                 self._net.update([((node.x,node.y),
                                   Node(parent=self,
                                        index=(node.x,node.y),
                                        can_stop=node.canStop,
                                        capacity=node.capacity,
                                        fare_probability=node.fareProb,
                                        traffic_cap=node.maxTraffic,
                                        traffic_in=node.tSrc,
                                        traffic_out=node.tSink,
                                        farefile=self._fareFile,
                                        **node.fareArgs))
                                   for node in nodes])
                 return
          print("Invalid nodes argument to add to Networld graph: not a list or tuple")

      # takes a list of streetDefs and adds it to the network as outgoing links. Interpolate
      # will create a series of interstitial nodes from the source to the destination. 
      def addEdges(self, edges, interpolate=False):

          # validate edge list
          if isinstance(edges, list) or isinstance(edges, tuple):
             try:
                 invalid = next(edge for edge in edges if not isinstance(edge, streetDef))
                 print("Invalid edge list to add to NetWorld graph: at least one object is not a streetDef")
                 return
             except StopIteration:
                 # in the interpolation case, we create additional interstitial nodes between junctions
                 if interpolate:
                    for edge in edges:
                        if edge.nodeA not in self._net:
                           raise ValueError("Node {0} does not exist in the map".format(edge.nodeA))
                        if edge.nodeB not in self._net:
                           raise ValueError("Node {0} does not exist in the map".format(edge.nodeB))
                        # extract source and destination-adjacent nodes, because we might exit each node
                        # along a slightly different angle than the direct origin-destination path
                        src = self._net[edge.nodeA]
                        dst = self._net[edge.nodeB]
                        srcExitx = src.index[0]
                        srcExity = src.index[1]
                        dstExitx = dst.index[0]
                        dstExity = dst.index[1]
                        '''
                        validate source-destination indices by edge direction. The key point here is that
                        an edge pointing in one direction cannot have a source 'in front' of it in either x or y
                        direction or a destination 'behind' it. Directions are indexed from 0 (North) through to
                        7 (North-West), so, for instance, if an edge had direction 5 (South-West), it could not
                        have a source node whose x index was less than the x-index of the destination, or whose
                        y index was greater than the y-index of the destination.
                        '''
                        if edge.dirA < 2 or edge.dirA > 6:
                           if src.index[1] <= dst.index[1]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(src.index, dst.index))
                           srcExity -= 1
                        if edge.dirA > 0 and edge.dirA < 4:
                           if src.index[0] >= dst.index[0]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(src.index, dst.index))    
                           srcExitx += 1
                        if edge.dirA > 4:
                           if src.index[0] <= dst.index[0]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(src.index, dst.index))
                           srcExitx -= 1
                        if edge.dirA > 2 and edge.dirA < 6:
                           if src.index[1] >= dst.index[1]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(src.index, dst.index))
                           srcExity += 1
                        if edge.dirB < 2 or edge.dirB > 6:
                           if dst.index[1] <= src.index[1]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(dst.index, src.index))  
                           dstExity -= 1
                        if edge.dirB > 0 and edge.dirB < 4:
                           if dst.index[0] >= src.index[0]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(dst.index, src.index))  
                           dstExitx += 1
                        if edge.dirB > 4:
                           if dst.index[0] <= src.index[0]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(dst.index, src.index))  
                           dstExitx -= 1
                        if edge.dirB > 2 and edge.dirB < 6:
                           if dst.index[1] >= src.index[1]:
                              raise ValueError("Road exit point from node {0} points away from destination node {1}".format(dst.index, src.index))  
                           dstExity += 1
                        nextNodeIdx = (srcExitx,srcExity)
                        # the trivial case where origin and destination are already adjacent can just be dealt with immediately
                        if nextNodeIdx[0] == dst.index[0] and nextNodeIdx[1] == dst.index[1]:
                           if self.addEdgeSegment(src, nextNodeIdx,edge.bidirectional) != dst:
                              raise KeyError("Indexed node ({0},{1}) should be the end junction {2}".format(nextNodeIdx[0],nextNodeIdx[1],dst.index))
                           return
                        # in other cases, note that the penultimate node may not exist. In the non-bidirectional case, this means we
                        # have to add it before stepping through the rest of the interpolated nodes, because again, its angle to the
                        # end node might be different than the straight-line origin-destination angle.
                        penultimateNodeIdx = (dstExitx,dstExity)
                        if not edge.bidirectional:
                           if penultimateNodeIdx not in self._net:
                              penultimateNode = Node(parent=self,
                                                     index=penultimateNodeIdx,
                                                     can_stop= True,
                                                     capacity=1,
                                                     farefile=self._fareFile,
                                                     **self.defaultFareGen)
                              self._net[penultimateNodeIdx] = penultimateNode
                           if self.addEdgeSegment(penultimateNode,(dst.index[0], dst.index[1]),False) != dst:
                              raise KeyError("Penultimate node {0} should be adjacent to end junction {1}".format(penultimateNode.index,dst.index))
                        # bidirectional cases can be just handled by the ordinary machinery.
                        else :
                              penultimateNode = self.addEdgeSegment(dst,penultimateNodeIdx,edge.bidirectional)
                        # insert the adjacent node from the origin
                        nextNode = self.addEdgeSegment(src, (srcExitx,srcExity), bidir=True)
                        # and then finally, step through between these 2 penultimate nodes to create the complete edge
                        while nextNode != penultimateNode:
                              nextNode = self.addEdgeSegment(nextNode, penultimateNodeIdx, bidir=True)

                 # when we don't have to interpolate, the process is a LOT simpler                           
                 else:
                    for edge in edges:
                        self._net[edge.nodeA].addNeighbour(self, self._net[edge.nodeB], edge.dirA)
                        if edge.bidirectional:
                           self._net[edge.nodeB].addNeighbour(self, self._net[edge.nodeA], edge.dirB)
          else:
             print("Invalid edges argument to add to Networld graph: not a list or tuple")

      # builds edges by adding one segment at a time, auto-computing entry and exit points.
      # start is the actual point from which we are going to add another segment. We assume
      # the start Node exists. end is NOT necessarily the next node, it is the ultimate endpoint
      # of this edge.
      def addEdgeSegment(self, start, end, bidir=True):
          # compute the position of the next point
          deltaX = end[0]-start.index[0]
          deltaY = end[1]-start.index[1]
          deltaTheta = math.atan2(deltaY, deltaX)
          xStep = round(math.cos(deltaTheta))
          yStep = round(math.sin(deltaTheta))
          nextIdx = (start.index[0]+xStep, start.index[1]+yStep)
          # which must not be out of the area of the network
          if nextIdx[0] > self.xSize-1 or nextIdx[0] < 0 or nextIdx[1] > self.ySize-1 or nextIdx[1] < 0:
             raise IndexError("Next node out of range: ({0},{1})".format(nextIdx[0], nextIdx[1]))
          # add it if it doesn't exist
          if nextIdx not in self._net:
             nextNode = Node(parent=self,
                             index=nextIdx,
                             can_stop=True,
                             capacity=2 if bidir else 1,
                             farefile=self._fareFile,
                             **self.defaultFareGen)
             self._net[nextIdx] = nextNode
          # find the exit directions from each node 
          startEgress = 0
          backEgress = 4
          if xStep > 0:
             if yStep > 0:
                startEgress = 3  
                backEgress = 7
             elif yStep < 0:
                startEgress = 1
                backEgress = 5
             else:
                startEgress = 2
                backEgress = 6
          elif xStep < 0:
             if yStep > 0:
                startEgress = 5
                backEgress = 1
             elif yStep < 0:
                startEgress = 7
                backEgress = 3
             else:
                startEgress = 6
                backEgress = 2
          else:
              if yStep > 0:
                 startEgress = 4
                 backEgress = 0
              else:
                 startEgress = 0
                 backEgress = 4
          # then add the links between nodes as necessary
          start.addNeighbour(self, self._net[nextIdx], startEgress)
          if bidir:
             self._net[nextIdx].addNeighbour(self, start, backEgress)
          return self._net[nextIdx]

      # places a taxi. Taxis can only come in at the boundaries of the service area, through established
      # roads. Like all other traffic, they have to gain admission to a Node; they're not automatically
      # allowed in
      def addTaxi(self, taxi, location):
          # taxis can only come in on the edges, and only if they're on duty
          if not taxi.onDuty:
             return (None, -1)
          ingressPoint = -1
          if location[0] == 0:
             if location[1] == 0:
                self._net[location].indicate(7,taxi)
                ingressPoint = 7
             elif location[1] == self.ySize-1:
                self._net[location].indicate(5,taxi)
                ingressPoint = 5
             else:
                self._net[location].indicate(6,taxi)
                ingressPoint = 6
          elif location[0] == self.xSize-1:
             if location[1] == 0:
                 self._net[location].indicate(1,taxi)
                 ingressPoint = 1
             elif location[1] == self.ySize-1:
                 self._net[location].indicate(3,taxi)
                 ingressPoint = 3
             else:
                 self._net[location].indicate(2,taxi)
                 ingressPoint = 2
          elif location[1] == 0:
             self._net[location].indicate(0,taxi)
             ingressPoint = 0
          elif location[1] == self.ySize-1:
             self._net[location].indicate(4,taxi)
             ingressPoint = 4
          else:
             return (None, -1)
          # the dispatcher ought to know a new taxi is available
          if self._dispatcher is not None:
             self._dispatcher.addTaxi(taxi)
          # a taxi just coming on duty has no predefined right of way
          self._taxis[taxi] = [(None, -1), (None, -1)]
          # but does get a traffic light indicator to allow it in
          return (self._net[location],ingressPoint)

      # adds a dispatcher. This is straightforward because the dispatcher doesn't need to have any physical location.
      # a dispatcher added using this method will supersede any previous dispatchers that have been there.
      def addDispatcher(self,dispatcher):
          # place the dispatcher in the world
          self._dispatcher = dispatcher
          # let it know the taxis that are already there,
          for taxi in self._taxis.keys():
              self._dispatcher.addTaxi(taxi)
          # give it the map of the service area,
          self._dispatcher.importMap(self.exportMap())
          # and any fares that happen to be waiting already
          for fare in self._fareQ.values():
              # we can also handle fares previously dispatched by another dispatcher
              if fare.taxi is not None:
                 self._dispatcher.handover(self, fare.origin, fare.destination, fare.callTime, fare.taxi, fare.price)
              else:
                 self._dispatcher.newFare(self, fare.origin, fare.destination, fare.callTime)

      #----------------------------------------------------------------------------------------------------------------

      #informational methods agents can use to interrogate various aspects of the world

      # accesses the node given an x,y coordinate
      def getNode(self, x, y):
          if (x,y) not in self._net:
             raise ValueError("Invalid node: ({0},{1})".format(x, y))
             return None
          return self._net[(x,y)]

      ''' this dumps out the complete map of the network. it is arranged as a nested dictionary. The
          outer dict is indexed by each node's (x,y) coordinate and there is one entry per node. Its
          inner dict is a map of the nodes to which the outer node connects directly, indexed by the
          destination (x,y) coordinate and giving a tuple of outward direction from the origin to the
          destination along with the distance to it.
      ''' 
      def exportMap(self):
          return dict([(node.index,
                        dict([((neighbour[1],neighbour[2]),
                               (neighbour[0], self.distance2Node(node,self._net[(neighbour[1],neighbour[2])])))
                              for neighbour in node.neighbours]))
                        for node in self._net.values()])

      # the next 2 functions are heuristics, that is, in some situations they will not be strictly
      # accurate. 
      # travel time between 2 nodes. If the nodes are directly connected this
      # will be an exact heuristic.
      def travelTime(self, origin, destination):
          # a None value from either node indicates coming in/going into 'The Void'.
          # regardless of situation, a taxi can always 'evaporate' into the void
          # effectively instantly
          if destination is None:
             return 0
          # but a taxi cannot simply materialise into a node; if the node is bunged
          # up with traffic, it will have to wait, potentially indefinitely
          if origin is None:
             if destination.traffic == destination.maxTraffic:
                return -1
             else:
                return 0
          # a travel time of -1 indicates an indefinite wait into the future. We are, in other words,
          # not getting anywhere anytime soon.
          if origin.traffic == origin.maxTraffic or destination.traffic == destination.maxTraffic:
             return -1
          # all other destinations take finite time to reach.
          else:
             return round((origin.traffic+destination.traffic+self.distance2Node(origin, destination))/2)

      # straight-line distance between 2 nodes. If the nodes are directly connected
      # this will be an exact heuristic
      def distance2Node(self, origin, destination):
          # give an invalid distance if the nodes were invalid
          if origin is None or destination is None:
             return -1
          return math.sqrt((destination.index[0]-origin.index[0])**2+(destination.index[1]-origin.index[1])**2)

      #_____________________________________________________________________________________________________________
      # methods called by world's members to execute coordinated actions
      
      '''methods generally called by Nodes
      '''
      
      # addTraffic is called by a Node and inserts traffic into another Node. We can
      # do this 'properly' on an event queue if desired
      def addTraffic(self, node):
          if node.index in self._trafficQ:
             self._trafficQ[node.index] +=1
          else:
             self._trafficQ[node.index] = 1
    
      # admitTaxi is called by a Node and registers it as the destination for Taxis.
      # admissionList is a dictionary of (direction, taxi) entries that give a
      # taxi the direction token needed to enter the Node.
      def admitTaxi(self, node, admissionList):
          for taxi in admissionList.items():
              if taxi[1] not in self._taxis:
                 raise ValueError("Node {0} tried to admit a taxi that does not exist".format(node.index))
              # off duty taxis do not need to be considered.
              if not taxi[1].onDuty:
                 self._taxis[taxi[1]][1] = (None, -1)
              # admit on-duty taxis with no existing admission
              if self._taxis[taxi[1]][1] == (None, -1):
                 self._taxis[taxi[1]][1] = (node, taxi[0])
              # A taxi with an existing admission, however, has it voided by a request to enter another Node
              # (and must try again).
              if self._taxis[taxi[1]][1][0] != node:
                 self._taxis[taxi[1]][1] = (None, -1)
                 
      # clearAdmission is called by a Node to release a taxi's admission when it has entered
      # the node.
      # --BUGFIX -- added 6 December 2020 
      def clearAdmission(self, node, taxi):
          if taxi in self._taxis and self._taxis[taxi][1][0] == node:
             # second field in each taxi's admission request is the node and direction to enter.
             newLoc = self._taxis[taxi][1]
             # it's now entered that node, so this becomes the first field in the admission request:
             # its current node and direction. Second field is cleared out.
             self._taxis[taxi] = [newLoc, (None, -1)]

      # insertFare is called by a Node on fare creation, adds it to the queue, and notifies
      # the Dispatcher
      def insertFare(self, node, fare):
          if node.index in self._fareQ:
             raise IndexError("Node {0} generated a new fare for one where a Fare is already waiting".format(node.index))
          # DEBUGME now destinations are generated by a FareGenerator object in a better way. Needs testing
          # generate a random valid destination. This isn't actually ideal, what we really want is a distance-dependent
          # distribution over the node pairs, but implementing a *fast* generator of such a form is not easy. Dictionaries,
          # furthermore, have in Python 3.x the subtlety that their values() can't be indexed, it returns an iterator
          # instead, so you have to turn it into a list. It may actually be faster to grab the keys(), turn those into
          # a list, then index the element in self._net.
          # destinationNode = node
          # while destinationNode == node or not destinationNode.canStop:
          #      destinationNode = list(self._net.values())[round(numpy.random.uniform(0,len(self._net)-1))]
          # fares will wait only for so long; a function of the distance to destination plus a gamma distribution 
          # maxWait = self.distance2Node(node, destinationNode)*10 + 5*numpy.random.gamma(2.0,1.0)
          # newFare = Fare(self, node, destinationNode, self._time, maxWait)
          # notify the Dispatcher, if any. If there is no Dispatcher yet, when it does come on-shift, it will
          # be notified of any pending Fares that it ought to dispatch, assuming they've not abandoned the attempt.
          # Dispatchers get no idea of how long a fare will wait! 
          if self._dispatcher is not None:
             self._dispatcher.newFare(self, fare.origin, fare.destination, fare.calltime)
          self._fareQ[fare.origin] = fare
          return

      def removeFare(self, fare):
          # if the fare wasn't collected, inform the dispatcher that they abandoned
          if not fare.enroute:
             if self._dispatcher is not None:
                self._dispatcher.cancelFare(self,
                                            fare.origin,
                                            fare.destination,
                                            fare.calltime)
          # both collected and abandoned fares disappear from the fare queue
          del self._fareQ[fare.origin]

      '''methods generally called by Dispatchers
      '''

      # broadcastFare is called by the Dispatcher, and issues a message to Taxis informing
      # them of a fare from origin to destination with a given price
      def broadcastFare(self, origin, destination, price):
          # small chance the fare has already given up, if the Dispatcher took too long to get around
          # to announcing it. If so, inform the Dispatcher
          if origin not in self._fareQ or self._fareQ[origin].destination != destination:
             return 0
          # let the fare know the price
          self._fareQ[origin].setPrice(price)
          onDuty = 0
          # inform the taxis,
          for taxi in self._taxis.keys():
              if taxi.onDuty:
                 onDuty +=1
                 taxi.recvMsg(taxi.FARE_ADVICE, **{'origin': origin, 'destination': destination, 'price': price})
          # and return how many taxis were advised
          return onDuty

      # allocateFare is called by the Dispatcher, and assigns a given fare to a given Taxi.
      def allocateFare(self, origin, taxi):
          # fare may have abandoned the attempt, while the taxi may have gone off-duty
          if origin not in self._fareQ or taxi not in self._taxis or not taxi.onDuty:
             return False
          self._fareQ[origin].assignTaxi(taxi)
          taxi.recvMsg(taxi.FARE_ALLOC, **{'origin': origin, 'destination': self._fareQ[origin].destination})

                      
      # cancelFare is called by the Dispatcher when a fare abandons their request, and informs any allocated
      # taxi so that they don't need to chase a nonexistent fare. This will also help taxis to estimate
      # fare-abandonment rates.
      def cancelFare(self, origin, taxi):
          if taxi not in self._taxis:
             return False
          # dispatchers *should* have a reference to the taxis they have allocated, and *should* check that
          # they're not off. But just in case, to prevent any spurious messages being sent...
          if taxi.onDuty:
             taxi.recvMsg(taxi.FARE_CANCEL, **{'origin': origin})

      '''methods generally called by Taxis
      '''

      # completeFare is called by the taxi and registers the fare as conducted to the destination
      # which then assigns the fare's price to the Taxi and Dispatcher.
      def completeFare(self, fare):
          self._dispatcher.recvPayment(self,fare.price*0.1)
          fare.taxi.recvMsg(fare.taxi.FARE_PAY, **{'amount': fare.price*0.9})
          # get rid of the fare's taxi allocation so that garbage collection doesn't have to worry
          # about back pointers. The taxi itself should already have got rid of the fare in the
          # completeFare call.
          fare.clear()

      # transmitFareBid is called by the taxi and notifies the Dispatcher that it it is willing
      # to accept the fare. It also notifies the Dispatcher of the taxi's location.
      def transmitFareBid(self, origin, taxi):
          self._dispatcher.fareBid(origin, taxi)

      #----------------------------------------------------------------------------------------------------------------
                 
      # runWorld operates the model. It can be run in single-stepping mode (ticks = 1), batch mode
      # (ticks = 0) or any number of step-aggregation modes (ticks > 1). In batch mode, live output
      # may be unreliable, if running in a separate thread.
      # added locking mechanism to protect live output. *Should* be reliable now. (BUGFIX 21 October 2025 ADR)
      def runWorld(self,ticks=0,outputs=None,outlock=None):
          if outputs is None:
             outputs = {}
             outlock = threading.Lock()
          ticksRun = 0
          while (ticks == 0 or ticksRun < ticks) and (self.runTime == 0 or self._time < self.runTime):
                print("Current time in the simulation world: {0}".format(self._time))
                outlock.acquire()
                if 'time' in outputs:
                   outputs['time'].append(self._time)
                outlock.release()
                # really simple recording of fares: just where there are fares still waiting. More
                # sophisticated recording including price information and enroute fare information,
                # could easily be added, e.g. by making the fare output a list of fare objects.
                outlock.acquire()
                if 'fares' in outputs:
                   for fare in self._fareQ.values():
                       if fare.origin in outputs['fares']:
                          outputs['fares'][fare.origin][self._time] = fare.calltime
                       else:
                          outputs['fares'][fare.origin] = {self._time: fare.calltime}
                outlock.release()
                # go through all the nodes and update the time tick
                for node in self._net.values():
                    node.clockTick(self)
                    # we can output live traffic information if we want. Or possibly other
                    # parameters of a node, depending on how much reporting is desirable. (With
                    # very large networks and lots of reporting, this could slow things considerably)
                    outlock.acquire()
                    if 'nodes' in outputs:
                        if node.index in outputs['nodes']:
                           outputs['nodes'][node.index][self._time] = node.traffic
                        else:
                           outputs['nodes'][node.index] = {self._time: node.traffic}
                    outlock.release()
                # next go through the (live) taxis
                for taxi in self._taxis.items():
                    if taxi[0].onDuty:
                       taxi[0].drive(taxi[1][1])
                       taxi[0].clockTick(self)
                       # similarly basic recording of taxis: just their current position, as long as they
                       # are on duty.
                       outlock.acquire()
                       if 'taxis' in outputs:
                          # taxi[1][0][0] is taxi[admission_request][current_pose][current_node] where
                          # current_pose is a (node, direction) pair. So this is just asking: is the taxi
                          # somewhere in the world?
                          if taxi[1][0][0] is not None:
                             if taxi[0].number in outputs['taxis']:
                                # outputs['taxis'][taxi[0].number][self._time] = taxi[0].currentLocation
                                outputs['taxis'][taxi[0].number][self._time] = taxi[1][0][0].index
                             else:
                                # outputs['taxis'][taxi[0].number] = {self._time: taxi[0].currentLocation}
                                outputs['taxis'][taxi[0].number] = {self._time: taxi[1][0][0].index}
                       outlock.release()
                    # an off-duty taxi can come on if it decides to (and will call addTaxi to add itself)
                    else:
                       taxi[0].comeOnDuty(self._time)
                # then run the dispatcher. With this ordering, taxis bidding for fares can always get
                # them allocated immediately (provided the dispatcher decides to do so). Taxis always
                # receive notice of potential fares for collection one clock after the fare first appeared
                # to the dispatcher. We can make this fully asynchronous if we wish with an event queue.
                if self._dispatcher is not None:
                   self._dispatcher.clockTick(self)
                # new traffic arrives last. Since we flow old traffic out of Nodes first, this gives
                # taxis the best chance to reach a Node, they shouldn't be helplessly stuck whilst
                # traffic flows around them.
                for node in self._trafficQ.items():
                    self._trafficQ[node[0]] -= self._net[node[0]].injectTraffic(self, node[1])
                # update the batch stepper
                self._time += 1
                ticksRun +=1                

 
