import numpy
import inspect
import faregenerator

'''
A Node represents any reachable point in the RoboUber world. Nodes can be thought of as lying on a
square grid with all the cardinal compass points being adjacent - i.e. the diagonals as well as 
the sides are accessible. A Node has a certain maximum occupancy (in terms of taxis). Any taxis
above the occupancy limit must wait until a space in the Node has been vacated. We allow more
than one occupant at some points to prevent there being a bottleneck at major junctions (nodes
with high in-degree). For simplicity we require that only one taxi per direction can ever be
present in a Node.
To simulate traffic, there are several flow control properties and methods built into a Node. 
Taxis must attempt to access a Node using indicate(). This signals intent to occupy the Node.
Taxis can abandon the attempt at any time using free(). When capacity becomes available, the
Node uses the traffic_light property to signal which taxi in the waiting list gains access.
Taxis must poll the traffic_light to see if they can enter, which they then do by using
occupy(). A taxi leaves a node using vacate() (which will, of course, only be possible if
the node it's moving to has space). 
There is also a traffic property, that the taxi can interrogate, which is a simple representation
of the overall (i.e. non-taxi) traffic in the Node. The amount of traffic in the node indicates
how many time steps are necessary to vacate the Node. If traffic reaches 5 (gridlock), the taxi
cannot vacate until traffic drops below 5.
There is a stop_allowed property. Fares can only be present at a node if stop is allowed. There
will only ever be one fare at a given stop (to minimise management of fare allocation). A taxi
can always opt simply to stop at a given location, but if stop is not allowed, it must not.

'''          
class Node:

      # class constructor. Most arguments are optional, but a Node must have an index
      # (a location) and a parent (a world it 'lives' in).
      
      def __init__(self,parent,index,can_stop=True,capacity=1,
                   fare_probability=None,traffic_cap=0,traffic_in=0,traffic_out=0,
                   N=None,S=None,E=None,W=None,NE=None,NW=None,SE=None,SW=None, farefile=None,
                   **fare_gen_args):

          # initialise neighbours in an adjacency list. This allows relatively simple computation
          # of turns; just compute the next positive or negative index, wrapping around.
          self._idx = index                        # index is the node's (x,y) tuple in the network
          self._neighbours = [N,NE,E,SE,S,SW,W,NW] # reachable neighbours (not necessarily symmetric)
          self._canStop = can_stop                 # taxis can stop here
          self._capacity = capacity                # max number of taxis that can be in this point 
          self._occupied = {}                      # dictionary of taxis at this point, indexed by current direction
          self._incoming = {}                      # dictionary of taxis attempting to enter this point
          self._traffic_light = 0                  # priority management for access. Indexes the direction with first priority
          self._trafficMax = traffic_cap           # _traffic point where the Node becomes locked
          self._trafficSrc = traffic_in            # amount of traffic automatically generated coming in per clock
          self._trafficSink = -traffic_out         # amount of traffic automatically removed per clock
          self._traffic = 0
          self._fare = None
          self._parent = parent

          # default the traffic capacity to a level of 8 (thus a junction can accept an input from all
          # sides before becoming overloaded)
          if self._trafficMax == 0:            
             self._trafficMax = 8
          # default fare generator just randomly produces a new fare with probability given by the size of the network
          if fare_probability is None:
             if self._parent.size == 0:
                # 1/1000 chance if there is no information on network size (approximately 1 call per day)
                self._fare_generator = faregenerator.FareGenerator(world=self._parent, parent_node=self, base_prob=0.001,
                                                                   fare_file=farefile, **fare_gen_args)
             else:
                # otherwise default probability would generate on average 1 call every 100 minutes for the service area
                self._fare_generator =  faregenerator.FareGenerator(world=self._parent, parent_node=self,
                                                                    base_prob=1/(10*self._parent.size()),
                                                                    fare_file=farefile, **fare_gen_args)
          else:
             # non-default fare generators can have all the normal types. If the types dict is not set (which is not required),
             # the generator will take the base probability but use the default fare generator otherwise.
             self._fare_generator = faregenerator.FareGenerator(self._parent, self, base_prob=fare_probability, fare_file=farefile, **fare_gen_args) 

      # properties of the Node that other objects can see
      
      @property
      def canStop(self):
          return self._canStop                
                
      @property
      def capacity(self):
          return self._capacity

      @property
      def haveSpace(self):
          return self._traffic < self._trafficMax

      @property
      def index(self):
          return self._idx

      @property
      def maxTraffic(self):
          return self._trafficMax
    
      # neighbours returns a list of non-None adjacent Nodes (i.e. accessible locations from
      # this Node) as tuples of the format (index, x, y) where index is the absolute directional
      # index in the _neighbours list
      @property
      def neighbours(self):
          return [(neighbour,
                   self._neighbours[neighbour].index[0],
                   self._neighbours[neighbour].index[1])
                  for neighbour in range(len(self._neighbours))
                  if self._neighbours[neighbour] is not None]
          
      @property
      def occupied(self):
          return len(self._occupied)

      @property
      def traffic(self):
          return self._traffic

      # methods generally called by the parent world

      # adds an adjoining node. Used in building the graph
      def addNeighbour(self, parent, neighbour, direction):
          if self._parent == parent:
             self._neighbours[direction] = neighbour

      ''' clockTick handles the core functionality of a node at each timestep. The function
          should be called by the Node's parent; there is a check that the calling object is
          indeed the parent. Within the timer tick the following things happen: taxis in the 
          incoming list are scheduled for access; traffic is flowed through (if traffic in 
          neighbouring Nodes is not blocking); and fares appear or disappear (the latter if
          it has been waiting too long)
      '''     
      def clockTick(self, parent):
          if self._parent == parent:
             # flow through traffic first, to give the node the best chance of allowing taxis in
             for neighbour in self._neighbours:
                 if neighbour is not None and neighbour.haveSpace and self._traffic > 0:
                    self._parent.addTraffic(neighbour)
                    self._traffic -= 1
                    self.injectTraffic(self._parent,self._trafficSink)
             # off-duty taxis which were here should be removed. Build a new list so
             # as not to do odd things in the _occupied dict
             offDuty = [taxi for taxi in self._occupied.items() if not taxi[1][0].onDuty]
             for taxi in offDuty:
                 del self._occupied[taxi[0]]
             # now deal with admitting taxis
             if len(self._incoming) > 0 and self._traffic < self._trafficMax and len(self._occupied) <= self._capacity:
                remaining = self._capacity - len(self._occupied)
                admitted = {}
                while remaining > 0 and len(admitted) < len(self._incoming):
                      # spin the traffic light until a taxi is found. This will
                      # admit taxis in fair round-robin fashion
                      while (self._traffic_light not in self._incoming and
                             self._traffic_light not in admitted):
                            self._traffic_light = (self._traffic_light+1) % 8
                      admitted[self._traffic_light] = self._incoming[self._traffic_light]
                      remaining -= 1
                self._parent.admitTaxi(self, admitted)
             # next deal with new fares appearing or abandoning
             if self._fare is not None:
                # fares won't wait forever for a ride
                if self._parent.simTime-self._fare.calltime > self._fare.maxWait:
                   self._parent.removeFare(self._fare)
                   self._fare = None
             # and will only hail a taxi in allowed stopping locations
             elif self._canStop:
                self._fare = self._fare_generator.generateFare()
                if self._fare is not None:
                   self._parent.insertFare(self, self._fare)
             # last thing to do is inject intrinsic traffic
             self.injectTraffic(self._parent,self._trafficSrc)

      # the world can interrogate the node for a given taxi to see if it is physically there.
      def hasTaxi(self, parent, taxi):
          if self._parent == parent:
             # quickly iterate with early stopping for the taxi in concern in the occupied dict
             try:
                 next(occ for occ in self._occupied.values() if occ[0] == taxi)
             except StopIteration:
                 # iterated through the whole dict without success. Taxi not there.
                 return False
             return True
       
      # add some traffic into the node. volume (the amount of traffic to inject) can be
      # negative, meaning traffic is removed from the node.
      def injectTraffic(self, parent, volume):
          if self._parent == parent:
             if self._traffic > self._trafficMax:
                return 0
             self._traffic += volume
             if self._traffic > self._trafficMax:
                excess = self._traffic - self._trafficMax
                self._traffic = self._trafficMax
                return volume-excess
             return volume

      # methods generally called by Taxis. Almost all of these are automated into the Taxi
      # machinery.

      ''' 
          ____________________________________________________________________________________________________________
          These methods allow the taxi to move between Nodes using drive. There are 3 steps.
          First, the taxi makes a decision to turn or continue straight through (if it can).
          Then, it must indicate to the Node that it is turning (and it can abandon the attempt
          if it waited too long).
          Finally, it vacates its old Node and occupies the new one.
      '''
              
      ''' turn requests an index for an adjacent node. This automatically indicates
          the direction change to the neighbour, and returns an (index, neighbour)
          tuple to the taxi showing which neighbour it should interrogate the traffic
          light for, and which direction the traffic light needs to show before it
          can access the node. 
          Direction indicates the relative egress point. 0 proceeds straight ahead, if this is possible.
          Positive directions index the nth possible left. Negative directions index the -nth
          possible right. A direction of None simply leaves the space without informing any
          adjacent space - this can be used e.g. to take down road works or other obstructions.
      '''
      def turn(self, directionIn, directionOut=-1):
          # if traffic is enabled, don't allow the taxi out of the space until a delay equivalent to the
          # traffic volume has elapsed. A traffic value of trafficMax indicates gridlock! In addition when the
          # taxi attempts to occupy a Node, the world controller will assign it a travel time to the
          # Node from the one it vacated. This means that even if traffic is disabled (self._traffic = 0)
          # the taxi takes finite time to traverse the Node.
          if self._traffic == self._trafficMax or self._parent.simTime-self._occupied[directionIn][1] < self._traffic:
             return (None, -1)
          relativeDirection = directionOut
          # default to straight ahead
          if relativeDirection  < 0:
             relativeDirection = (directionIn + 4) % 8
             # if directly straight ahead is not possible, the 2 diagonals could be sensible
             # provided only one is available. If both are available, we have an ambiguous fork
             # in the road.
             if self._neighbours[relativeDirection] is None:
                relativeDirection = (relativeDirection - 1) % 8
                alternativeDirection = (relativeDirection + 2) % 8
                if self._neighbours[relativeDirection] is None:
                   relativeDirection = alternativeDirection
                   if self._neighbours[relativeDirection] is None:
                      return (None, -1)
                elif self._neighbours[alternativeDirection] is not None:
                   return (None, -1)
          # otherwise the outward direction is indexed as appropriate
          # illegal direction; U-turns not allowed here or too few possible lefts/rights
          if self._neighbours[relativeDirection] is None:
             return (None, -1)     
          # otherwise indicate to the neighbour. In its world, incoming will be opposite outgoing
          newDirection = (relativeDirection + 4) % 8
          self._neighbours[relativeDirection].indicate(newDirection,self._occupied[directionIn][0])
          # and return what the new direction will be in the adjacent node, along with the node itself
          return (self._neighbours[relativeDirection], newDirection)

      # convenience function to avoid having taxis call the turn() method (which would be confusing in
      # this context) if they intend to proceed through straight ahead. This will fail if there is
      # no sensible 'straight ahead' direction.
      def continueThrough(self, directionIn):
          return self.turn(directionIn)

      # indicate requests access to the Node. direction is the incoming direction
      # as seen from the Node.
      def indicate(self, direction, occupant):
          self._incoming[direction] = occupant

      # abandon turns off an existing indication (e.g. if the taxi waited too long to gain admission)    
      def abandon(self, direction, occupant):
          if self._incoming[direction] == occupant:
             del self._incoming[direction]

      # claims the space. A vehicle can only occupy an available space.
      def occupy(self, direction, occupant, origin=None):
          # occupying a space takes finite time; it doesn't happen instantaneously.
          time2Occupy = self._parent.travelTime(origin,self)
          # all sorts of ways occupancy can fail: too many taxis already here; a taxi
          # has already taken the desired direction, the would-be occupant didn't
          # signal, a rogue taxi tried to swerve around the proper occupant, or this
          # node is backed up with traffic.
          if (len(self._occupied) == self._capacity or direction in self._occupied or
              direction not in self._incoming or self._incoming[direction] != occupant
              or time2Occupy < 0):
             print ("Taxi {0} can't occupy node: full".format(occupant.number))
             return (None, -1)
          self._occupied[direction] = (occupant,self._parent.simTime+time2Occupy)
          del self._incoming[direction]
          self._parent.clearAdmission(self,occupant) #BUGFIX clear from parent
          return (self, direction)

      # leaves the space if the occupant can. Returns the space occupied after the attempt
      # to vacate (which can fail). directionIn indexes which taxi is leaving. directionOut
      # is the index given by the target space that the taxi will be in when it enters the
      # space
      def vacate(self, directionIn, directionOut=None):
          # automatically vacate if there is no outward direction
          if directionOut is None:
             del self._occupied[directionIn]
             return (None, -1)
          # direction as seen from this Node is the diametric opposite of that seen
          # at the adjacent node
          relativeDirection = (directionOut + 4) % 8
          # occupy the neighbouring space if possible
          newSpace = self._neighbours[relativeDirection].occupy(directionOut,self._occupied[directionIn][0],self)
          if newSpace[0] is None and newSpace[1] == -1:
             return (self, directionIn)
          del self._occupied[directionIn]
          return newSpace

      '''--------------------------------------------------------------------------------------------------------
         These next methods deal with collecting and dropping off Fares.
      '''

      # taxis call this function to pick up the fare. The fare's pickUp routine will register
      # with the world controller that it has been collected by the calling taxi. A fare may
      # have since abandoned, in which case the attempt to pick up will fail. Otherwise a
      # taxi in the right location will get the fare returned to it.
      def pickupFare(self, direction=None):
          if direction in self._occupied:
             fare = self._fare
             if fare is not None and fare.taxi == self._occupied[direction][0]:
                self._fare = None
                fare.pickUp(self._occupied[direction][0])
                return fare
          return None
             
      # taxis call this function to drop off the fare. This will notify the world controller
      # that the fare has been conducted to the destination (and the controller will allocate
      # payments accordingly)
      def dropoffFare(self, fare, direction):
          if direction in self._occupied:
             # fares will only alight at their actual destination (naturally)
             if fare is not None and fare.destination == self._idx and self._occupied[direction][1] >= self._parent.simTime:
                fare.dropOff()
                self._parent.completeFare(fare)
                return True
          return False
