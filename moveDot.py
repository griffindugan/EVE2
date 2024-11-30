"""

Author: Griffin Dugan
Brandeis Memory & Cognition (Wingfield) Lab
EVE2 Dot Moving Test Program

Description: This program is designed to be an un-eyetracked version of the EVE2 main program. 
The program will display a red dot that will move to different locations on the screen. 
The program will then, when ready, move the dot to different locations on the screen. 
The program will move the dot to a location on the screen and wait for the participant. 
The program will continue to move the dot to different locations on the screen until the program is closed.
"""

import pygame
import pygame.freetype
import sys
import random
import numpy as np
import math
from time import sleep
from threading import Thread

FULL_SCREEN = False
SCREEN_SIZE = WIDTH, HEIGHT = (1280, 720) if not FULL_SCREEN else (0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
CIRCLE_RADIUS = 30

RO = 0
ORDER = [
    # 0   9  21  29  15  14  27  18   2  35  13  39  12  10  31   7  30  17  33   1  26   6  19  38   8  24  11   4  25  34   5  28  16  40  22  32  20  23  37   3  36
    23, 28, 28, 29, 28, 30, 28, 27, 28, 29, 28, 28, 31, 25, 29, 30, 29, 29, 32, 31, 27, 31, 24, 26, 23, 26, 29, 26, 27, 34, 28, 27, 34, 31, 31, 32, 32, 34, 33, 29, 29
]

tracked = [
    # p y n  n  y  y  n  y  y n  y  n  y  y  n  y n  y  n  y n  y y  n  y n  y  y n  n  y n  y  n  n  n  y  n  n  y n
    True, True, False, False, True, True, False, True, True, False, True, False, True, True, False, True, False, True, False, True, False, True, True, False, True, False, True, True, False, False, True, False, True, False, False, False, True, False, False, True, False
]

                



# # Initialization
# ROset = False
# while not ROset:
#     try:
#         number = input("Running Order? (1 or 2) ")
#         if number in ["1", "2"]:
#             RO = int(number)-1
#             ROset = True
#         else: print("You must input a number between 1 and 2.")
#     except ValueError:
#         print("You must input a number.")
RO = 1


pygame.init()
screen = pygame.display.set_mode(size=SCREEN_SIZE) if not FULL_SCREEN else pygame.display.set_mode(SCREEN_SIZE, pygame.FULLSCREEN)
if FULL_SCREEN: SCREEN_SIZE = WIDTH, HEIGHT = screen.get_size()
pygame.display.set_caption('aaaaaaaaaaaaa')
fps = pygame.time.Clock()
FONT = pygame.freetype.SysFont("helvetica-neue", 38)

CENTRE = (round(WIDTH/2), round(HEIGHT/2)) # centre by default is 400, 300
hGridSlice, wGridSlice = round(HEIGHT/6), round(WIDTH/6)
GRID = np.array([[(wGridSlice, hGridSlice*5), (wGridSlice*3, hGridSlice*5), (wGridSlice*5, hGridSlice*5)], # grid[width, height]
                [(wGridSlice, hGridSlice*3), (wGridSlice*3, hGridSlice*3), (wGridSlice*5, hGridSlice*3)],
                [(wGridSlice, hGridSlice),   (wGridSlice*3, hGridSlice),   (wGridSlice*5, hGridSlice)]])

# Ball setup
ball_pos = [GRID[0,1,0], GRID[0,1,1]]
framesPerSecond = 60
pixelsPerFrame = 60
base_speed = framesPerSecond * pixelsPerFrame # pixels per second 
stopBall, run, tracking = True, 0, True

def travel(x, y):
    global stopBall
    # determine the difference in current position to new position
    start_pos = np.array(ball_pos)
    end_pos = np.array([x, y])

    # Displacement
    delta = end_pos - start_pos

   # Aspect ratio (16:9) correction factors
    aspect_ratio_x = 16
    aspect_ratio_y = 9

    # Adjust for aspect ratio by normalizing components
    scaled_delta_x = delta[0] * aspect_ratio_y
    scaled_delta_y = delta[1] * aspect_ratio_x

    # Recompute distance with scaled deltas
    distance = np.sqrt(scaled_delta_x**2 + scaled_delta_y**2)

    total_frames = int(distance / pixelsPerFrame)

    # Normalize the movement per frame
    velocity_x = scaled_delta_x / distance * pixelsPerFrame
    velocity_y = scaled_delta_y / distance * pixelsPerFrame

    # Calculate positions for each frame
    positions = []
    for frame in range(total_frames + 1):
        # Update positions based on original delta but normalized
        pos_x = start_pos[0] + velocity_x * frame / aspect_ratio_y
        pos_y = start_pos[1] + velocity_y * frame / aspect_ratio_x
        positions.append(np.array([pos_x, pos_y]))

    # Ensure we end at the exact end position in the last frame
    positions[-1] = end_pos

    # # Output the positions
    # for i, pos in enumerate(positions):
    #     print(f"Frame {i + 1}: Position = {pos.astype(int)}")

    startPos, endPos = ball_pos, [x, y]
    diff = (x - ball_pos[0],round((9/16)*(y - ball_pos[1]))) # position differences

    magni = math.sqrt((diff[0]**2)+(diff[1]**2))

    # determine which axis needs more time
    # time = (abs(diff[0]/base_speed), abs(diff[1]/base_speed)) # determining amount of time it takes to get to each position
    time = (magni/(pixelsPerFrame*framesPerSecond))
    vx, vy = (diff[0])/(time*framesPerSecond) if diff[0] != 0 else 0, (diff[1])/(time*framesPerSecond) if diff[1] != 0 else 0
    # vx, vy = vxI*framesPerSecond, vyI*framesPerSecond
    # # determine speed of slower axis
    # if time[0] < time[1]: # scale y to x
    #     ball_speeds = (base_speed/framesPerSecond if diff[0]>0 else -1*(base_speed/framesPerSecond), 
    #                    (9/16)*(diff[1]/(time[0]*framesPerSecond)))
    #     tot_time = time[0]
    # else: # scale x to y
    #     ball_speeds = ((16/9)*(diff[0]/(time[1]*framesPerSecond)),
    #                    base_speed/framesPerSecond if diff[1]>0 else -1*(base_speed/framesPerSecond))
    #     tot_time = time[1]

    # ball_speeds = ((9/16)*ball_speeds[0], ball_speeds[1])


    # determine locations based on speeds
    frames = math.ceil(time*framesPerSecond)
    ball_positions = np.zeros((frames,2),dtype=np.int64) # empty array for each frame
    # print(f"\n\nCur Location: ({ball_pos[0]},{ball_pos[1]})\nEnd Location: ({x},{y})\nDiffs: {diff}\nTimes: {time}\nSpeeds:{ball_speeds}\n\n")
    for i in range(frames):
        if stopBall: break
        if i == frames-1: ball_positions[i,:] = (x, y) # on last frame
        else: ball_positions[i,:] = (ball_pos[0]+((i+1)*vx), # x locations 
                                     ball_pos[1]+((i+1)*vy)) # y locations
    
    # return list of locations 
    return(positions)

def update_ball():
    global stopBall, ball_pos
    end = ball_pos
    while end == ball_pos:
        height, width = random.choice([0,1,2]),random.choice([0,1,2])
        # height,width = 1, 0
        end = GRID[height,width].tolist()
    # call travel for list of locations
    locations = travel(end[0], end[1])

    # plot said locations 
    for i, (x, y) in enumerate(locations):
        if stopBall: break
        ball_pos[0] = x
        ball_pos[1] = y
        render_ball()
        True
    True
        # sleep(1)
    # ball_pos = [GRID[0,1,0], GRID[0,1,1]]
    # sleep(1)

def update_screen():
    phrase = "Please let the experimenter know you are ready."
    bounds = FONT.get_rect(phrase)
    screen.fill(BLACK)
    FONT.render_to(screen, (CENTRE[0]-(bounds.width/2),(hGridSlice*2)-(bounds.height/2)), phrase, WHITE)
    pygame.draw.line(screen, WHITE, [CENTRE[0]-30, CENTRE[1]], [CENTRE[0]+30, CENTRE[1]], 5)
    pygame.draw.line(screen, WHITE, [CENTRE[0], CENTRE[1]-30], [CENTRE[0], CENTRE[1]+30], 5)
    pygame.display.update()
    fps.tick(framesPerSecond)

def update():
    ball_pos[0] += random.choice([-5,0,5])
    ball_pos[1] += random.choice([-5,0,5])


def render_ball():
    screen.fill(BLACK)
    pygame.draw.circle(screen, RED, ball_pos, CIRCLE_RADIUS, 0)
    pygame.display.update()
    fps.tick(framesPerSecond)

def play_ball():
    global stopBall, tracking
    while not stopBall:
        if tracking: update_ball()
        else: render_ball()


def main():
    global stopBall, run, ball_pos, tracking, RO
    paused = False
    playing = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # if event.type == pygame.KEYUP:
            #     if event.key == pygame.K_SPACE:
            #         paused = not paused
            if event.type == pygame.MOUSEBUTTONUP and not playing and not paused:
                playing = True
        if not paused and playing:
            if run == 0: tracking = True
            else:
                tracking = tracked[run] if RO == 0 else not tracked[run]
            stopBall = False
            play_ball()
            # t = Thread(target=play_ball)
            # t.start()
            # sleep(ORDER[run])
            stopBall, playing, ball_pos = True, False, [GRID[0,1,0], GRID[0,1,1]]
            print(f"Run {run} completed.")
            run += 1
            # t.join()

            
        if not paused and not playing:
            update_screen()
            # break
            # render()

# travel(800, 500)
main()