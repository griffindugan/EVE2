"""

Author: Griffin Dugan
Brandeis Memory & Cognition (Wingfield) Lab
EVE2 Main Program


Description: This program shows a dot moving on the screen, while an eye tracker tracks the participant's focus point on the dot.
While this happens, the program plays an audio file and records the camera, which is expected to be recording the audiometer's volume number.
"""

# Variables that depend on the current scenario
dummy_mode = True
FULL_SCREEN = False
camera_number = 0



import pygame
import pygame.freetype
import sys
import random
import numpy as np
import math
import os
from time import sleep
from threading import Thread
import threading

from pydub import AudioSegment 
from pydub.playback import play #_play_with_pyaudio 
import cv2

# from eyeLinkFramework import *

import pylink
from pylink.eyeLinkFramework import *
import sys
from pygame.locals import *


# Generic necessary variables
SCREEN_SIZE = WIDTH, HEIGHT = (1024, 768) if not FULL_SCREEN else (0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
GREY = (128, 128, 128)
CIRCLE_RADIUS = 30

RO = 0

# Duration of Seconds of each passage
ORDER = [
    # 9  21  29  15  14  27  18   2  35  13  39  12  10  31   7  30  17  33   1  26   6  19  38   8  24  11   4  25  34   5  28  16  40  22  32  20  23  37   3  36
    28, 28, 29, 28, 30, 28, 27, 28, 29, 28, 28, 31, 25, 29, 30, 29, 29, 32, 31, 27, 31, 24, 26, 23, 26, 29, 26, 27, 34, 28, 27, 34, 31, 31, 32, 32, 34, 33, 29, 29, 30
]

# Whether the ball should move during each passage
tracked = [
    # y n  n  y  y  n  y  y n  y  n  y  y  n  y n  y  n  y n  y y  n  y n  y  y n  n  y n  y  n  n  n  y  n  n  y n
    True, False, False, True, True, False, True, True, False, True, False, True, True, False, True, False, True, False, True, False, True, True, False, True, False, True, True, False, False, True, False, True, False, False, False, True, False, False, True, False
]

# Passage numbers
runningOrder = np.array([
    #p y n  n  y  y  n  y  y n  y  n  y  y  n  y n  y  n  y n  y y  n  y n  y  y n  n  y n  y  n  n  n  y  n  n  y n
    0,9,21,29,15,14,27,18,2,35,13,39,12,10,31,7,30,17,33,1,26,6,19,38,8,24,11,4,25,34,5,28,16,40,22,32,20,23,37,3,36
])

# setting running order
ROset = False
while not ROset and not dummy_mode:
    try:
        number = input("Running Order? (1 or 2) ")
        if number in ["1", "2"]: # I think this should replace the try catch
            RO = int(number)-1
            ROset = True
        else: print("You must input a number that's either 1 or 2.")
    except ValueError:
        print("You must input a number.")
if dummy_mode: RO = 0 # if dummymode, just run base RO

# Initialise pygame
pygame.init()

# create folders for eyetracker
edf_fname, id = init_files() 

# make the folder for video
video_folder = os.path.join("results", id, f"video_files")
if not os.path.exists(video_folder):
    makeFolder(video_folder)

# intialise tracker
if dummy_mode: et = eyeLink(id=id, folders={"f":edf_fname})
else: et = eyeLink(id=id, address="100.1.1.1", folders={"f":edf_fname})


# Initialization of pygame
fps = pygame.time.Clock() # FPS counter
FONT = pygame.freetype.SysFont("helvetica-neue", 38) # base font

# initiate calibration of tracker
et.init_calibration(fullscreen=FULL_SCREEN, screensize=SCREEN_SIZE)
SCREEN_SIZE = WIDTH, HEIGHT = et.win.screensize # screen size based on created window size

# create grid for ball
CENTRE = (round(WIDTH/2), round(HEIGHT/2)) # centre of screen
# hGridSlice, wGridSlice = round(HEIGHT/6), round(WIDTH/6) # 1/6 slices for the grid
# GRID = np.array([[(wGridSlice, hGridSlice*5), (wGridSlice*3, hGridSlice*5), (wGridSlice*5, hGridSlice*5)], # grid[width, height]
#                 [(wGridSlice, hGridSlice*3), (wGridSlice*3, hGridSlice*3), (wGridSlice*5, hGridSlice*3)],
#                 [(wGridSlice, hGridSlice),   (wGridSlice*3, hGridSlice),   (wGridSlice*5, hGridSlice)]])

# Create a 9x9 grid based on screen size
rows, cols = 9, 9
x_spacing = WIDTH // (cols + 1)  # Divide width by number of columns + 1 for spacing
y_spacing = HEIGHT // (rows + 1)  # Divide height by number of rows + 1 for spacing

# Generate grid coordinates
x_coords = np.linspace(x_spacing, WIDTH - x_spacing, cols)
y_coords = np.linspace(y_spacing, HEIGHT - y_spacing, rows)
GRID = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)


# Ball setup
ball_pos = [GRID[40,0], GRID[40,1]] # starts in centre
framesPerSecond = 60 # fps
pixelsPerFrame = 20 # base speed
base_speed = framesPerSecond * pixelsPerFrame # pixels per second 
stopBall, run, tracking = True, 0, True # whether the ball should be moving/running, the trial index, and whether the ball should move or stay stationary
stopTrial = False

# Show the task instructions
task_msg = "In the task, please shift gaze to follow the RED dot\n" + \
    "You may press the SPACEBAR to end a trial\n" + \
    "or press ESC to if you need to quit the task early\n"
if dummy_mode:
    task_msg = task_msg + "\nNow, Press ENTER to start the task"
else:
    task_msg = task_msg + "\nNow, Press ENTER to calibrate tracker"

# start calibration of tracker
et.calibrate(task_msg)

#
# Controls
#

def run_controls() -> None:
    """Runs a control trial."""
    global et, run, stopBall, playing, ball_pos, tracking, control, practice # Probably don't need to global the variables
    if run < 2: tracking = False
    else: tracking = True

    screen = et.win.dis # defining for easy access 
    et.start_trial(run, "Control Tracked" if tracking else "Control Stationary") # start trial


    # draw a reference grid on the Host PC screen
    et.clearHostScreen(colour="GREY")
    et.drawHostLine(pos1=(et.win.width/2.0 - 350, et.win.height/2.0,), 
                    pos2=(et.win.width/2.0 + 350, et.win.height/2.0), 
                    colour="BLACK")
    et.drawHostLine(pos1=(et.win.width/2.0, et.win.height/2.0 - 350), 
                    pos2=(et.win.width/2.0, et.win.height/2.0 + 350), 
                    colour="BLACK")


    # drift check
    et.driftCheck(GRID[40,0],GRID[40,1])

    # wait for participant to be ready
    wait_msg = "When you are ready, press SPACEBAR to begin the trial."
    et.win.show_message(wait_msg, BLACK, GREY)
    et.win.wait_key([K_SPACE])

    # start recording
    et.record()

    # clearing screen
    et.clearDVScreen()

    # Starting up the trial
    stopBall, stopTrial = False, False
    t = Thread(target=control_duration) # sending the audio and recording to a different thread to process concurrently.
    t.start() # start the thread right as the trial begins
    trialLocations, endLocations = play_trial() # begin trial
    stopBall, playing, ball_pos = True, False, [GRID[40,0], GRID[40,1]] # reset the trial variables
    print(f"Control run {run} completed.")
    t.join() # close the thread


    # clear the screen
    screen.fill(GREY)
    pygame.display.flip()
    et.clearDVScreen()

    # Saving variable
    trial_info = [
        f"!V TRIAL_VAR tracking {tracking}",
        f"!V TRIAL_VAR duration {30}"
    ]

    # Stop recording
    et.stopRecording(trial_info=trial_info)

    
    run += 1
    if run == 4: 
        run = 0
        control, practice = False, True

def control_duration() -> None:
    """Sleeps for control duration."""
    global run, stopBall, stopTrial
    totalTime = 0
    while totalTime < 30:
        if stopTrial: break
        sleep(0.1)
        totalTime += 0.1
    # sleep(30) # sleep for 30 sec
    stopBall = True



#
# Practice
#

def run_practices() -> None:
    global et, run, stopBall, playing, ball_pos, tracking, control, practice # Probably don't need to global the variables
    tracking = True

    screen = et.win.dis # defining for easy access 
    et.start_trial(run+4, "Practice Tracked" if tracking else "Practice Stationary") # start trial


    # draw a reference grid on the Host PC screen
    et.clearHostScreen(colour="GREY")
    et.drawHostLine(pos1=(et.win.width/2.0 - 350, et.win.height/2.0), 
                    pos2=(et.win.width/2.0 + 350, et.win.height/2.0), 
                    colour="BLACK")
    et.drawHostLine(pos1=(et.win.width/2.0, et.win.height/2.0 - 350), 
                    pos2=(et.win.width/2.0, et.win.height/2.0 + 350), 
                    colour="BLACK")


    # drift check
    et.driftCheck(GRID[40,0],GRID[40,1])

    # wait for participant to be ready
    wait_msg = "When you are ready, press SPACEBAR to begin the trial."
    et.win.show_message(wait_msg, BLACK, GREY)
    et.win.wait_key([K_SPACE])

    # start recording
    et.record()

    # clearing screen
    et.clearDVScreen()

    # Starting up the trial
    stopBall, stopTrial = False, False
    t = Thread(target=practice_duration) # sending the audio and recording to a different thread to process concurrently.
    t.start() # start the thread right as the trial begins
    trialLocations, endLocations = play_trial() # begin trial
    stopBall, playing, ball_pos = True, False, [GRID[40,0], GRID[40,1]] # reset the trial variables
    print(f"Practice run {run} completed.")
    t.join() # close the thread


    # clear the screen
    screen.fill(GREY)
    pygame.display.flip()
    et.clearDVScreen()

    # Saving variable
    trial_info = [
        f"!V TRIAL_VAR tracking {tracking}",
        f"!V TRIAL_VAR duration {23}"
    ]

    # Stop recording
    et.stopRecording(trial_info=trial_info)

    
    run += 1
    if run == 3: 
        run = 0
        practice = False

def practice_duration() -> None:
    global ORDER, run, stopBall, id, camera_number # needed variables
    passageNumber = 0 # defining passage number for audio

    PATH = os.path.join(f"Stimuli/P{passageNumber}.wav") 
    passage = AudioSegment.from_wav(PATH) # grabbing the audio file

    cap = cv2.VideoCapture(camera_number) # defining capture device

    cap.set(cv2.CAP_PROP_FPS, 42) # defining FPS
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # defining Width/Height of the camera.
    codec = cv2.VideoWriter_fourcc("m","p","4","v") # defining the mp4 codec.
    recording_flag, frames = True, 0  # we are transitioning from not recording to recording

    t = Thread(target=play, args=(passage,)) # Setting up the threading for successful playing of the audio.

    # Beginning recording
    outPATH = os.path.join("results", id, f"video_files/Passage{passageNumber}.mp4")
    output = cv2.VideoWriter(outPATH, codec, 30, size,1) 

    t.start() # Begin audio at the same time as beginning recording.
    while True: 
        if frames >= (30*(passage.duration_seconds)) or not recording_flag or stopTrial: # stop recording once time has ended
            print("stopping recording") # TODO: For Final, cleanup print
            
            # Stopping recording and releasing the output to be saved.
            output.release()
            cap.release() 
            break # breaking out of the loop            

        # Reading and Writing Frames
        ret, frame = cap.read() 
        if ret == True: output.write(frame)
        else: print("end?") # TODO: figure out necessity?
        frames +=1
    stopBall = True # stop the ball from moving



#
# Trials
#

def run_trial() -> None:
    """Runs a trial."""
    global et, run, stopBall, playing, ball_pos, tracking, stopTrial # Probably don't need to global the variables
    tracking = tracked[run] if RO == 0 else not tracked[run] # determine tracking based on trial index and tracked list

    screen = et.win.dis # defining for easy access 
    et.start_trial(run+7, "Tracked" if tracking else "Stationary") # start trial


    # draw a reference grid on the Host PC screen
    et.clearHostScreen(colour="GREY")
    et.drawHostLine(pos1=(et.win.width/2.0 - 350, et.win.height/2.0), 
                    pos2=(et.win.width/2.0 + 350, et.win.height/2.0), 
                    colour="BLACK")
    et.drawHostLine(pos1=(et.win.width/2.0, et.win.height/2.0 - 350), 
                    pos2=(et.win.width/2.0, et.win.height/2.0 + 350), 
                    colour="BLACK")


    # drift check
    et.driftCheck(GRID[40,0],GRID[40,1])

    # wait for the participant to be ready
    wait_msg = "When you are ready, press SPACEBAR to begin the trial."
    et.win.show_message(wait_msg, BLACK, GREY)
    et.win.wait_key([K_SPACE])

    # start recording
    et.record()

    # clearing screen
    et.clearDVScreen()

    # Starting up the trial
    stopBall, stopTrial = False, False
    t = Thread(target=play_duration) # sending the audio and recording to a different thread to process concurrently.
    t.start() # start the thread right as the trial begins

    trialLocations, endLocations = play_trial() # begin trial
    stopBall, playing, ball_pos = True, False, [GRID[40,0], GRID[40,1]] # reset the trial variable
    t.join() # close the thread


    # clear the screen
    screen.fill(GREY)
    pygame.display.flip()
    et.clearDVScreen()

    # Saving variable
    trial_info = [
        f"!V TRIAL_VAR tracking {tracking}",
        f"!V TRIAL_VAR duration {ORDER[run]}"
    ]

    print(f"Run {run} completed.")
    run += 1

    # locations = []
    # print(trialLocations)
    # print()
    # for i in endLocations:
    #     locations.append(i[0])
    #     print(i)
    #     print()
    # print(locations)

    # Stop recording
    et.stopRecording(trial_info=trial_info)

def trial_duration() -> None:
    """Sleeps for trial duration."""
    global ORDER, run, stopBall
    sleep(ORDER[run])
    stopBall = True

def play_duration() -> None:
    """Plays correct audio for trial duration."""
    global ORDER, run, stopBall, runningOrder, id, camera_number, stopTrial # needed variables
    passageNumber = runningOrder[run] # defining passage number for audio

    PATH = os.path.join(f"Stimuli/P{passageNumber}.wav") 
    passage = AudioSegment.from_wav(PATH) # grabbing the audio file

    cap = cv2.VideoCapture(camera_number) # defining capture device

    cap.set(cv2.CAP_PROP_FPS, 42) # defining FPS
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) # defining Width/Height of the camera.
    codec = cv2.VideoWriter_fourcc("m","p","4","v") # defining the mp4 codec.
    recording_flag, frames = True, 0  # we are transitioning from not recording to recording

    t = Thread(target=play, args=(passage,)) # Setting up the threading for successful playing of the audio.

    # Beginning recording
    outPATH = os.path.join("results", id, f"video_files/Passage{passageNumber}.mp4")
    output = cv2.VideoWriter(outPATH, codec, 30, size,1) 

    t.start() # Begin audio at the same time as beginning recording.
    while True: 
        if frames >= (30*(passage.duration_seconds)) or not recording_flag or stopTrial: # stop recording once time has ended
            print("stopping recording") # TODO: For Final, cleanup print
    
            # Stopping recording and releasing the output to be saved.
            output.release()
            cap.release() 
            break # breaking out of the loop

        # Reading and Writing Frames
        ret, frame = cap.read() 
        if ret == True: output.write(frame)
        else: print("end?") # TODO: figure out necessity?
        frames +=1
    stopBall = True # stop the ball from moving

def playPassage(passage):
    global stopTrial
    player = _play_with_pyaudio(passage)
    while player.is_active():
        if stopTrial:
            player.stop_stream()
            break


def play_trial() -> pygame.event:
    """Runs the trial movement of dot.

    :return: Error if error occurs
    :rtype: None or eyeLink.error/event
    """
    global et, run, tracking, stopTrial
    screen, tracker = et.win.dis, et.et # easier to use variable names
    inProgress, totalLocations, endLocations = False, [], [] # function variables
    while not stopBall: # while the audio is playing
        if tracking: # if it's a tracking (moving ball) passage
            endX, endY = determineEnd() # determine end point
            locations = travel(endX, endY) # determine points along the path
            totalLocations.append(locations) # add them to total path
            endLocations.append((endX, endY))
            frameCount = 0
            

            for i, (x, y) in enumerate(locations): # for each location, move it
                if stopBall: break # check if audio has stopped midway

                if et.checkDisconnect(): return et.error # check for disconnect from eyetracker
                
                # check for keyboard events
                for ev in pygame.event.get():
                    if (ev.type == KEYDOWN) and (ev.key == K_b): # Abort a trial if "b" is pressed
                        print("SKIPPED TRIAL")
                        tracker.sendMessage("trial_skipped_by_user")
                        
                        # clear the screen
                        screen.fill(GREY)
                        pygame.display.flip()
                        
                        stopTrial = True
                        et.win.abort() # abort trial
                        return pylink.SKIP_TRIAL

                    # Terminate the task if ESCAPE
                    if (ev.type == KEYDOWN) and (ev.key == K_ESCAPE):
                        tracker.sendMessage("terminated_by_user")
                        et.terminate()
                        return pylink.ABORT_EXPT

                # draw the target
                screen.fill(GREY)
                ball_pos[0] = x
                ball_pos[1] = y
                pygame.draw.circle(screen, RED, ball_pos, CIRCLE_RADIUS, 0)
                pygame.display.flip()
                fps.tick(framesPerSecond)
                time = pygame.time.get_ticks()
                frameCount += 1

                # saving the interest area of the ball at each frame
                if frameCount == 1: # if first frame of movement
                    if not inProgress: # if it is the literal first frame of movement this entire trial
                        # send a message to mark movement onset
                        tracker.sendMessage("TARGET_ONSET")
                        
                        # save the Interest Area info following movement onset
                        ia_pars = (-1 * (0),
                                -1 * (time) + 1,
                                int(GRID[40,0] - CIRCLE_RADIUS),
                                int(GRID[40,1] - CIRCLE_RADIUS),
                                int(GRID[40,0] + CIRCLE_RADIUS),
                                int(GRID[40,1] + CIRCLE_RADIUS))
    
                        # start time
                        movement_start = time
                        inProgress = True

                        ia_msg = "{} {} CIRCLE 1 {} {} {} {} TARGET\n".format(*ia_pars)

                        et.create_ia(run, ia_msg, newTrial=True)
                    else: # if it's just the first frame, this movement path
                        # send a message to mark movement onset
                        tracker.sendMessage("TARGET_ONSET")

                        # save the Interest Area info following movement onset
                        ia_pars = (-1 * (pre_frame_time - movement_start), 
                                -1 * (time - movement_start) + 1,
                                int(pre_x - CIRCLE_RADIUS),
                                int(pre_y - CIRCLE_RADIUS),
                                int(pre_x + CIRCLE_RADIUS),
                                int(pre_y + CIRCLE_RADIUS))
                        
                        ia_msg = "{} {} CIRCLE 1 {} {} {} {} TARGET\n".format(*ia_pars)

                        # create interest area and open folder
                        et.create_ia(run, ia_msg)
                else: # otherwise, just save interest area
                    # send a message to mark movement onset
                    tracker.sendMessage("TARGET_ONSET")
                    # save the Interest Area info following movement onset
                    ia_pars = (-1 * (pre_frame_time - movement_start), # ticks as of previous frame
                            -1 * (time - movement_start) + 1, # ticks as of this frame
                            int(pre_x - CIRCLE_RADIUS), # point locations of the circle
                            int(pre_y - CIRCLE_RADIUS),
                            int(pre_x + CIRCLE_RADIUS),
                            int(pre_y + CIRCLE_RADIUS))
                    
                    ia_msg = "{} {} CIRCLE 1 {} {} {} {} TARGET\n".format(*ia_pars)

                    # create interest area and open folder
                    et.create_ia(run, ia_msg)

                # log the target position after each screen refresh
                tar_pos_msg = "!V TARGET_POS target {}, {} 1 0".format(*ball_pos)
                target_goal_msg = f"BALL LOCATION {endX}, {endY}"
                tracker.sendMessage(tar_pos_msg)
                tracker.sendMessage(target_goal_msg)              # send over another message to request Data Viewer to draw the pursuit target when visualizing the data
                et.clearDVScreen()
                tar_msg = "!V FIXPOINT 255 50 50 255 50 50 {} {} {} {}".format(*ball_pos, CIRCLE_RADIUS*2, CIRCLE_RADIUS*2)
                tracker.sendMessage(tar_msg)

                # keeping track of target position
                pre_frame_time = time
                pre_x = ball_pos[0]
                pre_y = ball_pos[1]
                tracker.sendMessage("TARGET_OFFSET")
            
        else: # otherwise, just keep the ball stationary
            frameCount = 0
            if stopBall: break

            if et.checkDisconnect(): return et.error # check for disconnect from tracker
            
            # check for keyboard events
            for ev in pygame.event.get():
                if (ev.type == KEYDOWN) and (ev.key == K_ESCAPE): # Abort a trial if "ESCAPE" is pressed
                    tracker.sendMessage("trial_skipped_by_user")
                    
                    # clear the screen
                    screen.fill(GREY)
                    pygame.display.flip()
                    
                    et.win.abort() # abort trial
                    return pylink.SKIP_TRIAL

                if (ev.type == KEYDOWN) and (ev.key == K_c): # Terminate the task if Ctrl-c
                    if ev.mod in [KMOD_LCTRL, KMOD_RCTRL, 4160, 4224]:
                        tracker.sendMessage("terminated_by_user")
                        et.terminate()
                        return pylink.ABORT_EXPT

            # draw the target
            screen.fill(GREY)
            pygame.draw.circle(screen, RED, ball_pos, CIRCLE_RADIUS, 0) # does not change location
            pygame.display.flip()
            fps.tick(framesPerSecond)
            time = pygame.time.get_ticks()
            frameCount += 1

            if frameCount == 1: # on first frame, mark onset
                # send a message to mark movement onset
                tracker.sendMessage("TARGET_ONSET")
                movement_start = time
                pre_frame_time = time
            
            # save the Interest Area info following movement onset
            ia_pars = (-1 * (pre_frame_time - movement_start),
                    -1 * (time - movement_start) + 1,
                    int(GRID[40,0] - CIRCLE_RADIUS),
                    int(GRID[40,1] - CIRCLE_RADIUS),
                    int(GRID[40,0] + CIRCLE_RADIUS),
                    int(GRID[40,1] + CIRCLE_RADIUS))
            
            ia_msg = "{} {} CIRCLE 1 {} {} {} {} TARGET\n".format(*ia_pars)

            # create interest area and open folder
            if frameCount == 1: et.create_ia(run, ia_msg, newTrial=True)
            else: et.create_ia(run, ia_msg)

            # log the target position after each screen refresh
            tar_pos_msg = "!V TARGET_POS target {}, {} 1 0".format(*ball_pos)
            target_goal_msg = f"BALL LOCATION {CENTRE[0]}, {CENTRE[1]}"
            tracker.sendMessage(tar_pos_msg)
            tracker.sendMessage(target_goal_msg)  

            # send over another message to request Data Viewer to draw the pursuit target when visualizing the data
            et.clearDVScreen()
            tar_msg = "!V FIXPOINT 255 0 0 255 0 0 {} {} 50 50".format(*ball_pos)
            tracker.sendMessage(tar_msg)

            # keep track of target position
            pre_frame_time = time
            tracker.sendMessage("TARGET_OFFSET")
    return totalLocations, endLocations

def determineEnd() -> tuple[int, int]:
    """Determines end location for the dot at the current position.

    :return: End locations
    :rtype: tuple[int, int]
    """
    global ball_pos
    end = ball_pos # setting end to start position
    while end == ball_pos: # while end location and start position are the same, find a new end position
        # height, width = random.choice([0,1,2]),random.choice([0,1,2]) # randomly picks a column for each height and width
        end = random.choice(GRID) # randomly picks a point from the grid
        end = end.tolist()
        # end = GRID[height,width].tolist() # grids them in a list
    return end[0], end[1] # returns them as integers

def travel(x:int, y:int) -> list[list[int]]:
    """Determines locations from start point to end point of the dot's movement based on speed and distance.

    :param x: End X location
    :type x: int
    :param y: End Y location
    :type y: int
    :return: List of positions to get to end location
    :rtype: list[list[int]]
    """
    global stopBall, ball_pos
    # determine the difference in current position to new position
    diff = (x - ball_pos[0],y - ball_pos[1]) # position differences

    # determine which axis needs more time
    time = (abs(diff[0]/base_speed), abs(diff[1]/base_speed)) # determining amount of time it takes to get to each position

    # determine speed of slower axis
    if time[0] > time[1]: ball_speeds, tot_time = (base_speed/framesPerSecond if diff[0]>0 else -1*(base_speed/framesPerSecond), diff[1]/(time[0]*framesPerSecond)), time[0] # if y is faster
    else: ball_speeds, tot_time = (diff[0]/(time[1]*framesPerSecond),base_speed/framesPerSecond if diff[1]>0 else -1*(base_speed/framesPerSecond)), time[1] # if x is faster

    # determine locations based on speeds
    frames = math.ceil(framesPerSecond*tot_time)
    ball_positions = np.zeros((frames,2),dtype=np.int64) # empty array for each frame
    for i in range(frames):
        if stopBall: break
        if i == frames-1: ball_positions[i,:] = (x, y) # on last frame
        else: ball_positions[i,:] = (ball_pos[0]+((i+1)*ball_speeds[0]), # x locations 
                                     ball_pos[1]+((i+1)*ball_speeds[1])) # y locations
    
    # return list of locations 
    return(ball_positions)


def wait_for_trial() -> None:
    """Wait for next trial screen."""
    screen = et.win.dis # variable name shortener
    phrase = "Please let the experimenter know you are ready." # phrase to show on screen
    bounds = FONT.get_rect(phrase) # figure out the size of it
    screen.fill(GREY)
    FONT.render_to(screen, (CENTRE[0]-(bounds.width/2),(2*HEIGHT/6)-(bounds.height/2)), phrase, BLACK) # render the phrase correctly
    # create a cross in centre of screen
    pygame.draw.line(screen, BLACK, [CENTRE[0]-30, CENTRE[1]], [CENTRE[0]+30, CENTRE[1]], 5) 
    pygame.draw.line(screen, BLACK, [CENTRE[0], CENTRE[1]-30], [CENTRE[0], CENTRE[1]+30], 5)
    pygame.display.flip()
    fps.tick(framesPerSecond)

def trials_completed() -> None:
    """Trials completed screen."""
    # This is effectively wait_for_trial() but with less complication
    screen = et.win.dis
    phrase = "Tests completed."
    bounds = FONT.get_rect(phrase)
    screen.fill(GREY)
    FONT.render_to(screen, (CENTRE[0]-(bounds.width/2),(2*HEIGHT/6)-(bounds.height/2)), phrase, BLACK)
    pygame.display.flip()
    fps.tick(framesPerSecond)

def main():
    global stopBall, run, ball_pos, tracking, RO, practice, control
    playing, practice, control, ended = False, False, True, False # if actively playing a trial, if practice trial, or if control trial
    while True: # until program closed, run
        for event in pygame.event.get(): # determine quitting
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONUP and not playing: # click to start trial
                playing = True
        # showGrid()
        if playing: # run trial when playing
            if control == True:
                run_controls()
            elif practice == True:
                run_practices()
            else:
                run_trial()
            playing = False # then turn playing back to false
        if run == len(ORDER): # when all trials are done, show tests completed page
            ended = True
            trials_completed()
        if not playing and not ended: # when not playing, show waiting screen
            wait_for_trial()
        


main()