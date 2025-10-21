import inspect

'''
A Fare is a very simple object residing in the world, representing a passenger to be delivered to some
destination starting from some origin. Fares wait for a taxi for a finite amount of time, given by 
maxWait, which is not known to the dispatcher or taxis, then abandon their request. Fares 'live' in a
queue ordered by abandon time, which is popped at each time step and sent to the dispatcher as a cancelled
fare. Meanwhile, they likewise appear at their origin nodes at random times - an event also given to the
dispatcher. They have a few simple methods which do the more or less obvious things: collecting and
dropping them off and assigning a price.
'''
class Fare:

      def __init__(self, parent, origin, destination, call_time, wait_time, cost_ceiling=None):

         self._parent = parent
         self._origin = origin
         self._destination = destination
         self._callTime = call_time
         self._waitTime = wait_time
         self._taxi = None
         self._enroute = False
         self._price = 0
         self._maxPrice = cost_ceiling

      # board a taxi
      def pickUp(self, taxi):
          if self._taxi == taxi:
             self._enroute = True
             self._parent.removeFare(self)

      # alight from a taxi
      def dropOff(self):
          self._enroute = False
          self._origin = self._destination

      # inform the fare of the expected price for the ride
      def setPrice(self, price):
          ''' abandon the call if the asked-for price is exorbitant (default is if
            it exceeds 10x the expected travel time, otherwise the fare may set a
            maximum price), or if the situation on the ground is so gridlocked that
            no one is getting anywhere, any time soon.'''
          expectedTime2Dest = self._parent.travelTime(self._origin, self._destination)
          if ((expectedTime2Dest < 0) or
              (self._maxPrice is None and price > 10*expectedTime2Dest) or
              (self._maxPrice is not None and price > self._maxPrice)) :
             print("Fare ({0},{1}) abandoned because expectedTime2Dest was {2} and price was {3}".format(self.origin[0],self.origin[1],expectedTime2Dest, self._price))
             self._waitTime = 0
          # otherwise, the price to be paid is as quoted.
          self._price = price

      # clear gets rid of any references to objects so that garbage collection can
      # delete the object for sure.
      def clear(self):
          self._parent = None
          self._origin = None
          self._destination = None
          self._taxi = None

      # once a fare has accepted a bid, assign it to the winning bidder taxi    
      def assignTaxi(self, taxi):
          self._taxi = taxi

      # fare has boarded and is on their way to destination    
      @property
      def enroute(self):
          return self._enroute

      # where the fare wants to be collected from
      @property
      def origin(self):
          return self._origin.index

      # where the fare wants to go to
      @property
      def destination(self):
            return self._destination.index

      # when the fare made the initial request for a taxi
      @property
      def calltime(self):
          return self._callTime

      # longest amount of time fare will wait before abandoning call
      @property
      def maxWait(self):
          return self._waitTime

      # agreed price for the trip
      @property
      def price(self):
          return self._price

      # taxi to which this fare has been assigned
      @property
      def taxi(self):
          return self._taxi
    
