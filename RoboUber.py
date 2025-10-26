import pygame
import threading
import time
import math
#import numpy
import sys
import json
import copy
# the 3 Python modules containing the RoboUber objects
import networld
import taxi
import dispatcher
# the main parameters are in an editable file
from ruparams import *

# RoboUber itself will be run as a separate thread for performance, so that screen
# redraws aren't interfering with model updates.
def runRoboUber(worldX,worldY,runTime,stop,junctions=None,streets=None,interpolate=False,outputValues=None,oLock=None,**args):

   # initialise a random fare generator
   if 'fareProb' not in args:
      args['fareProb'] = 0.001

   # might have a file for recording fare types (to gather data for learning them)
   if 'fareFile' not in args:
      args['fareFile'] = None
      
      
   # create the NetWorld - the service area
   print("Creating world...")
   svcArea = networld.NetWorld(x=worldX,y=worldY,runtime=runTime,fareprob=args['fareProb'],jctNodes=junctions,edges=streets,interpolateNodes=interpolate,farefile=args['fareFile'])
   print("Exporting map...")
   svcMap = svcArea.exportMap()
   if 'serviceMap' in args:
      args['serviceMap'] = svcMap

   # create some taxis
   print("Creating taxis")
   taxi0 = taxi.Taxi(world=svcArea,taxi_num=100,service_area=svcMap,start_point=(20,0))
   taxi1 = taxi.Taxi(world=svcArea,taxi_num=101,service_area=svcMap,start_point=(49,15))
   taxi2 = taxi.Taxi(world=svcArea,taxi_num=102,service_area=svcMap,start_point=(15,49))
   taxi3 = taxi.Taxi(world=svcArea,taxi_num=103,service_area=svcMap,start_point=(0,35))

   taxis = [taxi0,taxi1,taxi2,taxi3]

   # and a dispatcher
   print("Adding a dispatcher")
   dispatcher0 = dispatcher.Dispatcher(parent=svcArea,taxis=taxis)

   # who should be on duty
   svcArea.addDispatcher(dispatcher0)

   # bring the taxis on duty
   print("Bringing taxis on duty")
   for onDutyTaxi in taxis:
       onDutyTaxi.comeOnDuty()

   threadRunTime = runTime
   threadTime = 0
   print("Starting world")
   while threadTime < threadRunTime:

         # if the program may be quitting, stop execution awaiting user decision (BUGFIX 26 Nov 2024 ADR)
         args['ackStop'].wait()
         # exit if 'q' has been pressed
         if stop.is_set():
            threadRunTime = 0
         else: 
            svcArea.runWorld(ticks=1, outputs=outputValues, outlock=oLock)
            if threadTime != svcArea.simTime:
               threadTime += 1
            time.sleep(1)        # change this value to speed up the simulation. Smaller times = faster runs

# file to record appearing Fares. You can use similar instrumentation to record just about anything else of interest
if recordFares:
   fareFile = open('./faretypes.csv', 'a')
   print('"{0}"'.format('FareType'), '"{0}"'.format('originX'), '"{0}"'.format('originY'),
         '"{0}"'.format('destX'), '"{0}"'.format('destY'), '"{0}"'.format('MaxWait'), '"{0}"'.format('MaxCost'),
         sep=',', file=fareFile)
else:
   fareFile = None

# event to manage a user exit, invoked by pressing 'q' on the keyboard
userExit = threading.Event()
userConfirmExit = threading.Event()
userConfirmExit.set() # enable the simulation thread (BUGFIX 26 Nov 2024 ADR)

# pygame initialisation. Only do this once for static elements.
pygame.init()
displaySurface = pygame.display.set_mode(size=displaySize,flags=pygame.RESIZABLE) # |pygame.SCALED arrgh...new in pygame 2.0, but pip install installs 1.9.6 on Ubuntu 16.04 LTS
backgroundRect = None
aspectRatio = worldX/worldY
if aspectRatio > 4/3:
   activeSize = (displaySize[0]-100, (displaySize[0]-100)/aspectRatio)
else:
   activeSize = (aspectRatio*(displaySize[1]-100), displaySize[1]-100)
displayedBackground=pygame.Surface(activeSize)
displayedBackground.fill(pygame.Color(255,255,255))
activeRect = pygame.Rect(round((displaySize[0]-activeSize[0])/2),round((displaySize[1]-activeSize[1])/2),activeSize[0],activeSize[1])

meshSize = ((activeSize[0]/worldX),round(activeSize[1]/worldY))

# create a mesh of possible drawing positions
positions = [[pygame.Rect(round(x*meshSize[0]),
                          round(y*meshSize[1]),
                          round(meshSize[0]),
                          round(meshSize[1]))
              for y in range(worldY)]
             for x in range(worldX)]
drawPositions = [[displayedBackground.subsurface(positions[x][y]) for y in range(worldY)] for x in range(worldX)]

# junctions exist only at labelled locations; it's convenient to create subsurfaces for them
jctRect = pygame.Rect(round(meshSize[0]/4),
                      round(meshSize[1]/4),
                      round(meshSize[0]/2),
                      round(meshSize[1]/2))
jctSquares = [drawPositions[jct[0]][jct[1]].subsurface(jctRect) for jct in junctionIdxs]

# initialise the network edge drawings (as grey lines)
for street in streets:
    pygame.draw.aaline(displayedBackground,
                       pygame.Color(128,128,128),
                       (round(street.nodeA[0]*meshSize[0]+meshSize[0]/2),round(street.nodeA[1]*meshSize[1]+meshSize[1]/2)),
                       (round(street.nodeB[0]*meshSize[0]+meshSize[0]/2),round(street.nodeB[1]*meshSize[1]+meshSize[1]/2)))
    
# initialise the junction drawings (as grey boxes)
for jct in range(len(junctionIdxs)):
    jctSquares[jct].fill(pygame.Color(192,192,192))
    # note that the rectangle target in draw.rect refers to a Rect relative to the source surface, not an
    # absolute-coordinates Rect.
    pygame.draw.rect(jctSquares[jct],pygame.Color(128,128,128),pygame.Rect(0,0,round(meshSize[0]/2),round(meshSize[1]/2)),5)

# redraw the entire image    
displaySurface.blit(displayedBackground, activeRect)
pygame.display.flip()

# which taxi is associated with which colour
taxiColours = {}
# possible colours for taxis: black, blue, green, red, magenta, cyan, yellow, white
taxiPalette = [pygame.Color(0,0,0),
               pygame.Color(0,0,255),
               pygame.Color(0,255,0),
               pygame.Color(255,0,0),
               pygame.Color(255,0,255),
               pygame.Color(0,255,255),
               pygame.Color(255,255,0),
               pygame.Color(255,255,255)]

# relative positions of taxi and fare markers in a mesh point
taxiRect = pygame.Rect(round(meshSize[0]/3),
                       round(meshSize[1]/3),
                       round(meshSize[0]/3),
                       round(meshSize[1]/3))

fareRect = pygame.Rect(round(3*meshSize[0]/8),
                       round(3*meshSize[1]/8),
                       round(meshSize[0]/4),
                       round(meshSize[1]/4))

# you can run for more than a day if desired.
for run in range(numDays):

   # create a dict of things we want to record
   outputValues = {'time': [], 'fares': {}, 'taxis': {}}

   outputLock = threading.Lock() # BUGFIX 21 October 2025 protect outputValues dict

   # create the thread that runs the simulation
   roboUber = threading.Thread(target=runRoboUber,
                               name='RoboUberThread',
                               kwargs={'worldX':worldX,
                                       'worldY':worldY,
                                       'runTime':runTime,
                                       'stop':userExit,
                                       'ackStop': userConfirmExit,
                                       'junctions':junctions,
                                       'streets':streets,
                                       'interpolate':True,
                                       'outputValues':outputValues,
                                       'oLock': outputLock,
                                       'fareProb':fGenDefault,
                                       'fareFile':fareFile})
   
   # curTime is the time point currently displayed
   curTime = 0

   # start the simulation (which will automatically stop at the end of the run time)
   roboUber.start()

   # this is the display loop which updates the on-screen output.
   # Changed loop condition to allow early termination of simulation thread BUGFIX 21 October 2025 ADR
   while curTime < runTime-1:

         # you can end the simulation by pressing 'q'. This triggers an event which is also passed into the world loop
         try:
             quitevent = next(evt for evt in pygame.event.get() if evt.type == pygame.KEYDOWN) # and evt.key == pygame.K_q)
             if quitevent.key == pygame.K_q:
                userConfirmExit.clear() # have the simulation thread wait for a user response before continuing (BUGFIX 26 Nov 2024 ADR)
                print("Really quit? Press Y to quit, any other key to ignore and keep running")
                while not userConfirmExit.is_set():
                   try:
                       quitevent = next(evt for evt in pygame.event.get() if evt.type == pygame.KEYDOWN) # poll for confirmation (BUGFIX 26 Nov 2024 ADR)
                       if quitevent.key == pygame.K_y:
                          userExit.set()  # notify the roboUber thread that we are exiting
                          userConfirmExit.set() # then allow it to do so         (BUGFIX 26 Nov 2024 ADR)
                          roboUber.join() # wait 'til the simulation thread ends (BUGFIX 26 Nov 2024 ADR)
                          fareFile.close()
                          pygame.quit()
                          sys.exit()
                       userConfirmExit.set() # if user didn't want to quit, resume simulation (BUGFIX 26 Nov 2024 ADR)
                   except StopIteration:
                       continue
         # event queue had no 'q' keyboard events. Continue.
         except StopIteration:
             #pygame.event.get()
             if 'time' in outputValues and len(outputValues['time']) > 0 and curTime != outputValues['time'][-1]:
                print("curTime: {0}, world.time: {1}".format(curTime,outputValues['time'][-1]))

                # naive: redraw the entire map each time step. This could be improved by saving a list of squares
                # to redraw and being incremental, but there is a fair amount of bookkeeping involved.
                displayedBackground.fill(pygame.Color(255,255,255))
         
                for street in streets:
                    pygame.draw.aaline(displayedBackground,
                                       pygame.Color(128,128,128),
                                       (round(street.nodeA[0]*meshSize[0]+meshSize[0]/2),round(street.nodeA[1]*meshSize[1]+meshSize[1]/2)),
                                       (round(street.nodeB[0]*meshSize[0]+meshSize[0]/2),round(street.nodeB[1]*meshSize[1]+meshSize[1]/2)))
    
                for jct in range(len(junctionIdxs)):
                    jctSquares[jct].fill(pygame.Color(192,192,192))
                    pygame.draw.rect(jctSquares[jct],pygame.Color(128,128,128),pygame.Rect(0,0,round(meshSize[0]/2),round(meshSize[1]/2)),5)
             
                # get fares and taxis that need to be redrawn. We find these by checking the recording dicts
                # for time points in advance of our current display timepoint. The nested comprehensions
                # look formidable, but are simply extracting members with a time stamp ahead of our
                # most recent display time. The odd indexing fare[1].keys()[-1] gets the last element
                # in the time sequence dictionary for a fare (or taxi), which, because of the way this
                # is recorded, is guaranteed to be the most recent entry.
                outputLock.acquire()
                faresToRedraw = dict([(rfare[0], dict([(time[0], time[1])
                                                      for time in rfare[1].items()
                                                      if time[0] > curTime]))
                                      for rfare in outputValues['fares'].items()
                                      if max(rfare[1].keys()) > curTime])
                                      #if sorted(list(fare[1].keys()))[-1] > curTime])
                outputLock.release()
                outputLock.acquire()
                taxisToRedraw = dict([(rtaxi[0], dict([(taxiPos[0], taxiPos[1])
                                                      for taxiPos in rtaxi[1].items()
                                                      if taxiPos[0] > curTime]))
                                      for rtaxi in outputValues['taxis'].items()
                                      if max(rtaxi[1].keys()) > curTime])
                                      #if sorted(list(taxi[1].keys()))[-1] > curTime])
                outputLock.release()

                # some taxis are on duty?
                if len(taxisToRedraw) > 0:
                   for rtaxi in taxisToRedraw.items():
                       # new ones should be assigned a colour
                       if rtaxi[0] not in taxiColours and len(taxiPalette) > 0:
                          taxiColours[rtaxi[0]] = taxiPalette.pop(0)
                       # but only plot taxis up to the palette limit (which can be easily extended)
                       if rtaxi[0] in taxiColours:
                          newestTime = max(rtaxi[1].keys())
                          # a taxi shows up as a circle in its colour
                          pygame.draw.circle(drawPositions[rtaxi[1][newestTime][0]][rtaxi[1][newestTime][1]],
                                             taxiColours[rtaxi[0]],
                                             (round(meshSize[0]/2),round(meshSize[1]/2)),
                                             round(meshSize[0]/3))
                   
                # some fares still awaiting a taxi?
                if len(faresToRedraw) > 0:
                   for rfare in faresToRedraw.items():
                       newestFareTime = max(rfare[1].keys())
                       # fares are plotted as orange triangles (using pygame's points representation which
                       # is relative to the rectangular surface on which you are drawing)
                       pygame.draw.polygon(drawPositions[rfare[0][0]][rfare[0][1]],
                                           pygame.Color(255,128,0),
                                           [(meshSize[0]/2,meshSize[1]/4),
                                            (meshSize[0]/2-math.cos(math.pi/6)*meshSize[1]/4,meshSize[1]/2+math.sin(math.pi/6)*meshSize[1]/4),
                                            (meshSize[0]/2+math.cos(math.pi/6)*meshSize[1]/4,meshSize[1]/2+math.sin(math.pi/6)*meshSize[1]/4)])
                   
                # redraw the whole map 
                displaySurface.blit(displayedBackground, activeRect)
                pygame.display.flip()

                # advance the time                           
                curTime += 1

   roboUber.join() # wait 'til the simulation thread ends (BUGFIX 26 Nov 2024 ADR)
   print("end of day: {0}".format(run))
# reached the end of the loop. Next day (or exit)
if fareFile is not None: # BUGFIX handle no open fare file to record ADR 21 October 2025
   fareFile.close()
pygame.quit()
sys.exit()




      
