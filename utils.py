import numpy as np
import RobotDART as rd

def create_grid(box_step_x=0.05, box_step_y=0.05):
    box_positions = []
    box_x_min = 0.3
    box_x_max = 0.7
    box_y_min = -0.4
    box_y_max = 0.4

    box_nx_steps = int(np.floor((box_x_max-box_x_min) / box_step_x))
    box_ny_steps = int(np.floor((box_y_max-box_y_min) / box_step_y))

    for x in range(box_nx_steps+1):
        for y in range(box_ny_steps+1):
            box_x = box_x_min + x * box_step_x
            box_y = box_y_min + y * box_step_y
            # if (np.linalg.norm([box_x, box_y]) < 1.):
            #     continue
            box_positions.append((box_x, box_y))

    return box_positions


def create_problems():
    cubes = ['red', 'green', 'blue']

    problems = []

    for cubeA in cubes:
        for cubeB in cubes:
            if cubeB == cubeA:
                continue
            for cubeC in cubes:
                if cubeC == cubeA or cubeC == cubeB:
                    continue
                problems.append([cubeA, cubeB, cubeC])
    
    return problems
