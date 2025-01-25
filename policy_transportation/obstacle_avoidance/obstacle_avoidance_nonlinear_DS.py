import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import math
import numpy as np

import matplotlib.pyplot as plt

from vartools.states import Pose
from vartools.dynamics import LinearSystem

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from nonlinear_avoidance.dynamics import SimpleCircularDynamics
from nonlinear_avoidance.multi_obstacle import MultiObstacle
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleAvoider
from nonlinear_avoidance.multi_obstacle_avoider import MultiObstacleContainer
from nonlinear_avoidance.arch_obstacle import create_arch_obstacle

from nonlinear_avoidance.multi_obstacle_container import plot_multi_obstacle_container


def integrate_trajectory(
    start_positions, velocity_functor,
    dt=0.01, it_max=200, abs_tol=1e-1,
    reduce_acceleration: bool = False):

    positions = np.zeros((start_positions.shape[0], it_max + 1))  # (2,201)
    
    positions[:, 0] = start_positions
    tmp_velocity = velocity_functor(positions[:, 0])
    for ii in range(it_max):
        velocity = velocity_functor(positions[:, ii])

        if np.linalg.norm(velocity) < abs_tol:
            return positions[:, : ii + 1]

        # Reduce velocity when going to far apart
        dotprod = np.dot(velocity, tmp_velocity)
        if not np.isclose(dotprod, 0):
            dotprod = dotprod / (np.linalg.norm(velocity) * np.linalg.norm(tmp_velocity))
        scaling = (1 + dotprod) / 2.0 + abs_tol
        scaling = min(1.0, scaling)

        # velocity = velocity * scaling
        tmp_velocity = velocity

        positions[:, ii + 1] = positions[:, ii] + velocity * dt
        # break
    return positions


def test_straight_system_with_tree(visualize=False):
    dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    container = MultiObstacleContainer()
    obstacle_tree = MultiObstacle(Pose(np.array([0, 0.0])))
    obstacle_tree.set_root(
        Cuboid(
            center_position=np.array([-2.0, 0]),
            axes_length=np.array([2.0, 3.0]),
            margin_absolut=0.0,
            distance_scaling=1.0))
    
    obstacle_tree.add_component(
        Cuboid(
            center_position=np.array([-2.0, 1.0]),
            axes_length=np.array([4.0, 1.0]),
            margin_absolut=0.0,
            distance_scaling=1.0,),
        reference_position=np.zeros(2),parent_ind=0)
    
    container.append(obstacle_tree)

    # MultiObstacleAvoider.create_with_convergence_dynamics() will only instantiate the class MultiObstacleAvoider() with given parameters
    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(
        obstacle_container=container,
        initial_dynamics=dynamics,
        create_convergence_dynamics=True)

    if visualize:
        x_lim = [-5, 3]
        y_lim = [-4, 4]

        n_resolution = 40
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(ax=ax, obstacle_container=obstacle_tree, x_lim=x_lim, y_lim=y_lim)

        plot_obstacle_dynamics(
            obstacle_container=obstacle_tree,
            dynamics=avoider.evaluate_sequence,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            attractor_position=dynamics.attractor_position)

    position = np.array([-0.28, 1.52])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[1] / velocity[0]) < 1e-1, "Parallel to border"

    position = np.array([0.4, 3.10])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([0.4, 3.13])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)

    position = np.array([-2.3, -1.55])
    velocity = avoider.evaluate_sequence(position)
    assert abs(velocity[0] / velocity[1]) > 1e2, "Expected to be parallel to wall."
    assert velocity[0] > 0.0

    position = np.array([-4.8, -1.8])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([-4.8, -1.795])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)

    # position = np.array([-2.01, -4.785])
    position = np.array([-4.3469387755, 3.0204081632])
    velocity1 = avoider.evaluate(position)
    position = np.array([-4.33, 3.0])
    velocity2 = avoider.evaluate(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)

    position = np.array([-4.76, -2.01])
    velocity1 = avoider.evaluate_sequence(position)
    position = np.array([-4.75, -2.01])
    velocity2 = avoider.evaluate_sequence(position)
    assert np.allclose(velocity1, velocity2, atol=1e-1)


#  currently working on this
def test_trajectory_integration(visualize=False):
    dynamics = SimpleCircularDynamics(pose=Pose(np.array([0.25, 0.0])),radius=0.4)  # this is the center and radius of limit cycle

    container = MultiObstacleContainer()

    obstacle_tree = MultiObstacle(Pose(np.array([0.0, 0.0])))  # this is the position of entire obstacle tree 

    obstacle_tree.set_root(
        Cuboid(center_position=np.array([0.0, -0.5]),  # this is the position of single obstacle from obstacle tree 
                axes_length=np.array([0.16, 0.16]),
                margin_absolut=0.025,
                distance_scaling=8.0))
    
    container.append(obstacle_tree)

    # MultiObstacleAvoider.create_with_convergence_dynamics() will only instantiate the class MultiObstacleAvoider() with given parameters
    avoider = MultiObstacleAvoider.create_with_convergence_dynamics(  
        obstacle_container=container,
        initial_dynamics=dynamics,
        # reference_dynamics=linearsystem(attractor_position=dynamics.attractor_position),
        create_convergence_dynamics=True)

    if visualize:
        x_lim = [-1, 1]
        y_lim = [-1, 1]

        n_resolution = 2
        figsize = (6, 5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_multi_obstacle_container(ax=ax, container=container, x_lim=x_lim, y_lim=y_lim) # this will only plot the obstacle

        # initial starting states
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_resolution),
            np.linspace(y_lim[0], y_lim[1], n_resolution))
        
        start_positions = np.array([xx.flatten(), yy.flatten()])
        
        for ii in range(start_positions.shape[1]):
            trajectory = integrate_trajectory(start_positions[:, ii], avoider.evaluate_sequence)  # velocity_func = avoider.evaluate_sequence

            color = "black"
            ax.plot(trajectory[0, :], trajectory[1, :], "-")
            ax.plot(trajectory[0, 0], trajectory[1, 0], "x", color=color)  # initial state
            ax.plot(trajectory[0, -1], trajectory[1, -1], "o", color=color)  # goal state


if (__name__) == "__main__":
    np.set_printoptions(precision=3)

    test_straight_system_with_tree(visualize=True)
    # test_trajectory_integration(visualize=True)

    plt.show()
