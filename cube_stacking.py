import numpy as np
import RobotDART as rd
import dartpy
import py_trees
from dataclasses import dataclass

from utils import create_grid, create_problems

# dataclass for checks object
@dataclass
class Checks:
    has_cube: bool
    at_cube: bool
    empty: bool
    at_zero: bool
    
    done:bool
    sim:bool

# Inits
steps = -1
Checks_status = Checks(False, False, True, False, False, False)

# Returns the Rotation matrix based on one of the cubes
def Active_Rotation(tf, pose_vec, axis):
    new_tf = tf.rotation()
    # check axis alignment
    #NOTE: the sign is not the usual for z rotations because there is a rotation misalignment of the robot base position
    if "x" in axis:
        new_tf[0][0] = -np.cos(pose_vec[2])
        new_tf[0][1] = -np.sin(pose_vec[2])
        new_tf[1][0] = -np.sin(pose_vec[2])
        new_tf[1][1] = np.cos(pose_vec[2])
    elif "y" in axis:
        new_tf[0][0] = -np.sin(pose_vec[2])
        new_tf[0][1] = np.cos(pose_vec[2])
        new_tf[1][0] = np.cos(pose_vec[2])
        new_tf[1][1] = np.sin(pose_vec[2])

    return new_tf

# Exactly the same as in examples
def damped_pseudoinverse(jac, l=0.01):
    m, n = jac.shape
    if n >= m:
        return jac.T @ np.linalg.inv(jac @ jac.T + l*l*np.eye(m))
    return np.linalg.inv(jac.T @ jac + l*l*np.eye(n)) @ jac.T

#ANCHOR: PI Controller
class PITask:
    def __init__(self, target, dt, Kp=10., Ki=0.1, Kd=1.):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._Ki = Ki
        self._Kd = Kd
        self._sum_error = 0
        self._prev_error = 0

    def set_target(self, target):
        self._target = target

    # Error Function for seperate rot and linear error
    def error(self, tf):
        rot_error = rd.math.logMap(self._target.rotation() @ tf.rotation().T)
        lin_error = self._target.translation() - tf.translation()
        return np.r_[rot_error, lin_error]
    # Update function
    def update(self, current):
        error_in_world_frame = self.error(current)
        self._sum_error = self._sum_error + error_in_world_frame * self._dt

        # err_dot =  (self.prev_error - error_in_world_frame) / self._dt
        # self._prev_error = error_in_world_frame
        # return self._Kp * error_in_world_frame + self._Ki * self._sum_error + self._Kd * err_dot
        
        return self._Kp * error_in_world_frame + self._Ki * self._sum_error

#ANCHOR: P controller
class P_Controller:
    def __init__(self, target, dt, Kp=10.):
        self._target = target
        self._dt = dt
        self._Kp = Kp
        self._sum_error = 0.

    def set_target(self, target):
        self._target = target

    #Error function for simple minus
    def error(self, current):
        error = self._target - current
        return error

    def update(self, current):
        error = self.error(current)
        return self._Kp * error

#ANCHOR: Behavior ReachTarget
class ReachTarget(py_trees.behaviour.Behaviour):
    def __init__(self, robot, tf_desired, type, goal_robot, dt, name="ReachTarget"):
        super(ReachTarget, self).__init__(name)

        self.robot = robot
        self.eef_link_name = "panda_ee"
        self.type = type
        # set the desired target stack (global): Call by Object Reference
        self.tf_desired_stack = tf_desired
        # dt
        self.dt = dt
        # Counter for Target Changes
        self.count = 0

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" %
                          (self.__class__.__name__))

    # Initialise runs every time the behavior is ticked and anytime the status is not running
    def initialise(self):
        self.logger.debug("%s.initialise()->init controller" %
                          (self.__class__.__name__))

        # Set Sub-Targets for trajectory

        #check if ending is near and set last position
        tf_current = self.robot.body_pose(self.eef_link_name)
        counter = 1
        if Checks_status.done == True and counter==1:
            counter +=1
            tf_current.set_translation([tf_current.translation()[0], tf_current.translation()[1], 5*box_size[2]])
            self.tf_desired_stack.append(tf_current)

        # Pop desired target from global stack
        self.tf = self.tf_desired_stack.pop()

        
        #NOTE: This behavior incomporates every type of movement of robot arm based on desired target. Each Movement incomporates three sub-movements with three sub-targets so that there is no collisions with nearby cubes
        # Sub-Target 1: Above existing position
        self.tf_target_1 = dartpy.math.Isometry3()
        self.tf_target_1.set_translation(
            [tf_current.translation()[0], tf_current.translation()[1], 3*box_size[2]])
        tf_current_vec = self.robot.body_pose_vec(self.eef_link_name)
        self.tf_target_1.set_rotation(self.tf.rotation())

        # Sub-Target 2: Above desired target 
        self.tf_target_2 = dartpy.math.Isometry3()
        self.tf_target_2.set_translation([self.tf.translation()[0], self.tf.translation()[
                                         1], self.tf.translation()[2] + 3*box_size[2]])
        self.tf_target_2.set_rotation(self.tf.rotation())

        # Sub-Target 3: Exact desired position
        self.tf_target_3 = dartpy.math.Isometry3()
        self.tf_target_3.set_translation(self.tf.translation())
        self.tf_target_3.set_rotation(self.tf.rotation())

        # Sub-Target 4: In case of last Target Reach
        self.tf_target_4 = dartpy.math.Isometry3()
        self.tf_target_4.set_translation([self.tf_target_3.translation()[0], self.tf_target_3.translation()[
                                         1], self.tf_target_3.translation()[2] + 3*box_size[2]])
        self.tf_target_4.set_rotation(self.tf.rotation())

        # Initialization for the controllers
        self.Kp = 2.  # Kp could be an array of 6 values
        self.Ki = 0.01  # Ki could be an array of 6 values
        self.controller_1 = PITask(self.tf_target_1, self.dt, self.Kp, self.Ki)

        # P controller has target 0 for 0. in position of finger in position vector
        self.controller_grab = P_Controller(0., self.dt, 1.)

        # Set a goal target for each subtarget for visualization
        goal_robot.set_base_pose(self.controller_1._target)

    def update(self):
        new_status = py_trees.common.Status.RUNNING

        # Get velocities from controller
        tf = self.robot.body_pose(self.eef_link_name)
        vel = self.controller_1.update(tf)

        # If error smaller than 1e-2 then change sub-target
        if np.linalg.norm(self.controller_1.error(tf)) < 1e-2 and Checks_status.done == False:
            self.count += 1

            if self.count == 1:
                self.controller_1._target = self.tf_target_2
                goal_robot.set_base_pose(self.controller_1._target)
            if self.count == 2:
                self.controller_1._target = self.tf_target_3
                goal_robot.set_base_pose(self.controller_1._target)
        elif Checks_status.done == True and Checks_status.sim == False:
            self.tf_desired_stack.append(self.tf_target_4)

        # Sub target in case tower is done
        if len(self.tf_desired_stack) == 0 and Checks_status.sim == False:
            print("So Close")
            
            Checks_status.sim = True


        # Jacobian in world frame
        jac = self.robot.jacobian(self.eef_link_name)  
        # Get damped pseudoinverse
        jac_pinv = damped_pseudoinverse(jac)
        # Get Robot commands
        cmd = jac_pinv @ vel

        # In case of possession of cube then activate second controller
        if Checks_status.has_cube:
            tf_grabbed = self.robot.positions()

            vel = self.controller_grab.update(tf_grabbed[7])
            cmds1 = [0., 0., 0., 0., 0., 0., 0., vel, 0.]

            # Get robot commands from both controllers
            cmd = cmd + cmds1
            self.robot.set_commands(cmd)
        else:
            self.robot.set_commands(cmd)

        # if error too small, report success
        err = np.linalg.norm(self.controller_1.error(tf))
        if err < 1e-3 and self.count >= 2 and Checks_status.done == False:
            new_status = py_trees.common.Status.SUCCESS
        # if error too small and tower done, report success and last movement (not used)
        elif err < 1e-3 and self.count >= 2 and Checks_status.done == True:
            new_status = py_trees.common.Status.SUCCESS
        
            
        # Set checks status and return Success, Failure or Running
        if new_status == py_trees.common.Status.SUCCESS:
            self.count = 0
            if self.type in "at_cube":
                Checks_status.at_cube = True
                Checks_status.at_zero = False
            if self.type in "at_zero":
                Checks_status.at_zero = True
                Checks_status.at_cube = False
            self.feedback_message = "Reached target"
            self.logger.debug("%s.update()[%s->%s][%s]" % (
                self.__class__.__name__, self.status, new_status, self.feedback_message))
        elif new_status == py_trees.common.Status.FAILURE:
            if self.type in "at_cube":
                Checks_status.at_cube = False
            if self.type in "at_zero":
                Checks_status.at_zero = False
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (
                self.__class__.__name__, self.status, self.feedback_message))
        else:
            if self.type in "at_cube":
                Checks_status.at_cube = False
            if self.type in "at_zero":
                Checks_status.at_zero = False
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (
                self.__class__.__name__, self.status, self.feedback_message))

        return new_status

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

#ANCHOR: Behavior Handle Cube
class HandleCube(py_trees.behaviour.Behaviour):
    def __init__(self, robot, type, target, name="HandleCube"):
        super(HandleCube, self).__init__(name)
        self.robot = robot
        self.target = target
        self.type = type
        self.counter = 0
        self.dt = dt

        self.logger.debug("%s.__init__()" % (self.__class__.__name__))

    def setup(self):
        self.logger.debug("%s.setup()->does nothing" %
                          (self.__class__.__name__))

    def initialise(self):
        self.logger.debug("%s.initialise()->does nothing" %
                          (self.__class__.__name__))
        
        # Initialize P controller
        self.Kp = 2.  # Kp could be an array of 6 values
        self.controller = P_Controller(self.target, self.dt, self.Kp)

    def update(self):
        new_status = py_trees.common.Status.RUNNING
        
        # Get positions
        tf_current = self.robot.positions()
        # get velocities from Controller
        vel = self.controller.update(tf_current[7])
        #Set robot Commands
        cmds1 = [0., 0., 0., 0., 0., 0., 0., vel, 0.]
        self.robot.set_commands(cmds1)

        # if error too small, report success
        err = self.controller.error(tf_current[7])

        # Depending on type of handling set different error thresholds and wait for 10 more ticks before changing behavior
        if abs(err) < 2.5e-2 and self.type in "grab":
            self.counter = self.counter + 1
            if self.counter > 10:
                new_status = py_trees.common.Status.SUCCESS

        if abs(err) < 1e-2 and self.type in "let":
            self.counter = self.counter + 1
            if self.counter > 10:
                new_status = py_trees.common.Status.SUCCESS
      
        # Set checks status and return Success, Failure or Running
        if new_status == py_trees.common.Status.SUCCESS:
            self.counter = 0
            if self.type in "grab":
                Checks_status.has_cube = True
                Checks_status.empty = False
            if self.type in "let":
                Checks_status.empty = True
                Checks_status.has_cube = False
            
            # Check if done
            if Checks_status.sim == True:
                    Checks_status.done = True
                    Checks_status.sim = False

            self.feedback_message = self.type + " Cube"
            self.logger.debug("%s.update()[%s->%s][%s]" % (
                self.__class__.__name__, self.status, new_status, self.feedback_message))

        elif new_status == py_trees.common.Status.FAILURE:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (
                self.__class__.__name__, self.status, self.feedback_message))
        else:
            self.feedback_message = "Error: {0}".format(err)
            self.logger.debug("%s.update()[%s][%s]" % (
                self.__class__.__name__, self.status, self.feedback_message))

        return new_status

    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

#ANCHOR: Behavior Check
class Check(py_trees.behaviour.Behaviour):
    def __init__(self, check, name="Check"):
        super(Check, self).__init__(name)
        self.check = check
        self.logger.debug("%s.__init__()" % (self.__class__.__name__))
    def setup(self):
        self.logger.debug("%s.setup()->does nothing" %
                          (self.__class__.__name__))
    def initialise(self):
        self.logger.debug("%s.initialise()->does nothing" %
                          (self.__class__.__name__))
    def update(self):
        new_status = py_trees.common.Status.RUNNING
        # Decide which check is handled
        if self.check in "has_cube":
            check = Checks_status.has_cube
        elif self.check in "at_cube":
            check = Checks_status.at_cube
        elif self.check in "empty":
            check = Checks_status.empty
        elif self.check in "at_zero":
            check = Checks_status.at_zero
        elif self.check in "done":
            check = Checks_status.done

        # Set checks status and return Success or Failure
        if check:
            new_status = py_trees.common.Status.SUCCESS
            self.feedback_message = "Good"
            self.logger.debug("%s.update()[%s->%s][%s]" % (
                self.__class__.__name__, self.status, new_status, self.feedback_message))
        else:
            new_status = py_trees.common.Status.FAILURE
            self.feedback_message = "Uh oh"
            self.logger.debug(
                "%s.update()[%s->%s]" % (self.__class__.__name__, self.status, self.feedback_message))
        return new_status
    def terminate(self, new_status):
        self.logger.debug(
            "%s.terminate()[%s->%s]" % (self.__class__.__name__, self.status, new_status))

#########################################################
dt = 0.001  # you are NOT allowed to change this
simulation_time = 1000.0  # you are allowed to change this
total_steps = int(simulation_time / dt)

#########################################################
# DO NOT CHANGE ANYTHING IN HERE
# Create robot
robot = rd.Franka(int(1. / dt))
init_position = [0., np.pi / 4., 0., -
                 np.pi / 4., 0., np.pi / 2., 0., 0.04, 0.04]
robot.set_positions(init_position)

max_force = 5.
robot.set_force_lower_limits(
    [-max_force, -max_force], ["panda_finger_joint1", "panda_finger_joint2"])
robot.set_force_upper_limits([max_force, max_force], [
                             "panda_finger_joint1", "panda_finger_joint2"])
#########################################################
robot.set_actuator_types("servo")  # you can use torque here
# robot.set_position_enforced(True)

#########################################################
# DO NOT CHANGE ANYTHING IN HERE
# Create boxes
box_positions = create_grid()

box_size = [0.04, 0.04, 0.04]

# Red Box
# Random cube position
red_box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[red_box_pt][0],
            box_positions[red_box_pt][1], box_size[2] / 2.0]
red_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                              0.9, 0.1, 0.1, 1.0], "red_box")

# Green Box
# Random cube position
green_box_pt = np.random.choice(len(box_positions))
while green_box_pt == red_box_pt:
    green_box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[green_box_pt][0],
            box_positions[green_box_pt][1], box_size[2] / 2.0]
green_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                                0.1, 0.9, 0.1, 1.0], "green_box")

# Blue Box
# Random cube position
box_pt = np.random.choice(len(box_positions))
while box_pt == green_box_pt or box_pt == red_box_pt:
    box_pt = np.random.choice(len(box_positions))
box_pose = [0., 0., 0., box_positions[box_pt][0],
            box_positions[box_pt][1], box_size[2] / 2.0]
blue_box = rd.Robot.create_box(box_size, box_pose, "free", 0.1, [
                               0.1, 0.1, 0.9, 1.0], "blue_box")
#########################################################

#########################################################
# PROBLEM DEFINITION
# Choose problem
problems = create_problems()
problem_id = np.random.choice(len(problems))
problem = problems[problem_id]

print('We want to put the', problem[2], 'cube on top of the', problem[1],
      'and the', problem[1], 'cube on top of the', problem[0], 'cube.')

# Get the Problem position names
position_0 = problem[0] + "_box"
position_1 = problem[1] + "_box"
position_2 = problem[2] + "_box"

#Create a shadow robot for visualizing targets
eef_link_name = "panda_ee"
tf_desired = robot.body_pose(eef_link_name)
vec_desired = robot.body_pose_vec(eef_link_name)

goal_robot = rd.Robot.create_ellipsoid(dims=[0.15, 0.15, 0.15], pose=vec_desired, color=[
                                       0., 1., 0., 0.5], ellipsoid_name="target")


# Set the position targets according to the problem
targets = []

#REVIEW: Edge Cases for axis orientation
# TODO: Add margins instead of points
# if red_box.base_pose_vec[3] == blue_box.base_pose_vec[3] or red_box.base_pose_vec[3] == green_box.base_pose_vec[0] or blue_box.base_pose_vec[3] == green_box.base_pose_vec[3]:
#     axis = "x"
#     x_axis = True
# if red_box.base_pose_vec[4] == blue_box.base_pose_vec[4] or red_box.base_pose_vec[4] == green_box.base_pose_vec[4] or blue_box.base_pose_vec[4] == green_box.base_pose_vec[4]:
#     axis = "y"
#     y_axis = True
# if x_axis and y_axis:
#     print("No posible rotation to solve it without collisions")

#TODO: Set it so it can take dynamic positions of each so that when it falls it goes to the new position to pick it up
# Get new rotation matrix from red cube
new_rotation = Active_Rotation(tf_desired, red_box.base_pose_vec(), "y")

# Set targets from the lists of local variables
#NOTE: Each target gets treated for a small translation a bit higher that the original target to avoid collisions between cubes
targets.append(locals()[position_0].base_pose())
targets[0].set_translation([targets[0].translation()[0], targets[0].translation()[
                           1], targets[0].translation()[2] + (2 * box_size[2]) + 0.01])
targets[0].set_rotation(new_rotation)

targets.append(locals()[position_2].base_pose())
targets[1].set_rotation(new_rotation)

targets.append(locals()[position_0].base_pose())
targets[2].set_translation([targets[2].translation()[0], targets[2].translation()[
                           1], targets[2].translation()[2] + box_size[2] + 0.01])
targets[2].set_rotation(new_rotation)

targets.append(locals()[position_1].base_pose())
targets[3].set_rotation(new_rotation)

#########################################################

#########################################################
# Create Graphics
gconfig = rd.gui.Graphics.default_configuration()
gconfig.width = 1280  # you can change the graphics resolution
gconfig.height = 960  # you can change the graphics resolution
graphics = rd.gui.Graphics(gconfig)

# Create simulator object
simu = rd.RobotDARTSimu(dt)
simu.set_collision_detector("fcl")  # you can use bullet here
simu.set_control_freq(100)
simu.set_graphics(graphics)
graphics.look_at((0., 4.5, 2.5), (0., 0., 0.25))
simu.add_checkerboard_floor()
simu.add_robot(robot)
simu.add_visual_robot(goal_robot)
simu.add_robot(red_box)
simu.add_robot(blue_box)
simu.add_robot(green_box)
#########################################################

# Behavior Tree
py_trees.logging.level = py_trees.logging.Level.DEBUG

# Create tree root
root = py_trees.composites.Parallel(
    name="Root", policy=py_trees.common.ParallelPolicy.SuccessOnOne())
# Create sequence node (for sequential targets)
sequence_one = py_trees.composites.Sequence(name="Sequence_one", memory=True)

# NOTE: 1) Selector Grab Cube
selector_grab_cube = py_trees.composites.Selector(
    name="Selector Grab Cube", memory=True)

# ANCHOR: Check Cube
check_has_cube = Check("has_cube", "Check Has_Cube")
selector_grab_cube.add_child(check_has_cube)

# ANCHOR: Sequence Grab
sequence_grab = py_trees.composites.Sequence(name="Sequence_grab", memory=True)
check_at_cube = Check("at_cube", "Check At_Cube")
grab_cube = HandleCube(robot, "grab", 0., "Grab Cube")
sequence_grab.add_child(check_at_cube)
sequence_grab.add_child(grab_cube)
selector_grab_cube.add_child(sequence_grab)

# ANCHOR: Go to Cube
go_to_cube = ReachTarget(robot, targets, "at_cube",
                         goal_robot, dt, "Reach Target Cube")
selector_grab_cube.add_child(go_to_cube)

sequence_one.add_child(selector_grab_cube)

# NOTE: 2) Selector Let Cube
selector_let_cube = py_trees.composites.Selector(
    name="Selector Let Cube", memory=True)

# ANCHOR: Check Empty
check_empty = Check("empty", "Check Empty")
selector_let_cube.add_child(check_empty)

# ANCHOR: Sequence Let
sequence_let = py_trees.composites.Sequence(name="Sequence_let", memory=True)
check_at_zero = Check("at_zero", "Check At_Zero")
let_cube = HandleCube(robot, "let", 0.04, "LetCube")
sequence_let.add_child(check_at_zero)
sequence_let.add_child(let_cube)
selector_let_cube.add_child(sequence_let)

# ANCHOR: Go to Zero
go_to_zero = ReachTarget(robot, targets, "at_zero",
                         goal_robot, dt, "Reach Target Zero")
selector_let_cube.add_child(go_to_zero)

sequence_one.add_child(selector_let_cube)

# Set main selector
selector_total = py_trees.composites.Selector(
    name="Selector Total", memory=True)

# Check if tower is done
check_at_zero = Check("done", "Check Done")

go_to_last_pos = ReachTarget(robot, targets, "at_last",
                         goal_robot, dt, "Reach Last Target")

sequence_End = py_trees.composites.Sequence(name="Sequence_End", memory=True)

sequence_End.add_child(check_at_zero)
sequence_End.add_child(go_to_last_pos)
# add them to the tree
selector_total.add_child(sequence_End)
selector_total.add_child(sequence_one)

root.add_child(selector_total)

# Render tree structure
py_trees.display.render_dot_tree(root)

# tick once
root.tick_once()

#########################################################


for step in range(total_steps):
    if (simu.schedule(simu.control_freq())):
        root.tick_once()
        # if Checks_status.done == True and Checks_status.sim == True:
        #     print("a little bit")
        #     steps = step + 10000
        #     Checks_status.sim = False
        pass
        
        # if step == steps:
        #     break

    if (simu.step_world()):
        break
