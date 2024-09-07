import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from tqdm import tqdm


def plot_vector_field(model, datax_grid, datay_grid, demo, surface):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))
    vel = model.predict(pos_array)
    u = vel[:, 0].reshape(dataXX.shape)
    v = vel[:, 1].reshape(dataXX.shape)
    
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    ax.set_aspect(1)
    ax.streamplot(dataXX, dataYY, u, v, density=2)
    ax.scatter(demo[:, 0], demo[:, 1], color=[1, 0, 0])
    ax.scatter(surface[:, 0], surface[:, 1], color=[0, 0, 0])

#################################################################################################

#  Implement the modulation matrix M as per equation (5) in the paper, This is a simplified version
def modulation_matrix_for_spherical(state, obstacle_center, obstacle_radius):
    M = np.zeros((state.shape[0],2,2))
    for i in range(state.shape[0]):
        q = state[i,:].reshape(2,1) - obstacle_center
        d = np.linalg.norm(q)
        n = q / d
        e = np.array([[-n[1,0]], [n[0,0]]])
        E = np.hstack((n,e))

        lambda_1 = 1 - np.power((obstacle_radius / d),2)
        lambda_2 = 1 + np.power((obstacle_radius / d),2)
        D = np.diag([lambda_1, lambda_2])
        M[i,:,:] = E @ D @ E.T
    return M

def rk4_step(model, current_state, obstacle_center, radius, dt):
    def f(model, Y, O_c, r):
        M = modulation_matrix_for_spherical(Y, O_c, r)  # shape (2,2)
        DS = model.predict(Y).T  # shape (2,1)
        return M[0] @ DS

    k1 = f(model, current_state, obstacle_center, radius).T
    k2 = f(model, current_state + 0.5 * dt * k1, obstacle_center, radius).T
    k3 = f(model, current_state + 0.5 * dt * k2, obstacle_center, radius).T
    k4 = f(model, current_state + dt * k3, obstacle_center, radius).T
    
    next_state = current_state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return next_state  # shape (1,2)

def plot_modified_vector_field_1(model,datax_grid,datay_grid,demo,surface,obstacle_center,radius):
    # For Entire State-Space
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)  #  both will have a shape of (100,100)
    pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))  # (10000,2)

    M = modulation_matrix_for_spherical(pos_array,obstacle_center,radius)  # shape (10000,2,2)

    vel = model.predict(pos_array)  # shape (10000,2)
    
    # Modulated velocity
    new_vel = np.squeeze(np.matmul(M, np.expand_dims(vel,axis=2)), axis=2)

    u = new_vel[:, 0].reshape(dataXX.shape)
    v = new_vel[:, 1].reshape(dataXX.shape)

    # For one Trajectory
    dt = 1
    current_state = demo[[0],:]  # shape (1,2)
    goal_state = demo[[-1],:]  # shape (1,2)
    new_trajectry = current_state
    # for _ in range(demo.shape[0]):
    while np.linalg.norm(goal_state - current_state) > 1:
        M1 = modulation_matrix_for_spherical(current_state, obstacle_center, radius)  # shape (10000,2,2)
        future_state = current_state.T + (M1[0] @ model.predict(current_state).T) * dt
        current_state = future_state.T
        new_trajectry = np.append(new_trajectry, current_state, axis=0)
        # print(np.linalg.norm(goal_state - current_state))

    # # For one Trajectory
    # dt = 1
    # current_state = demo[[0],:]  # shape (1,2)
    # goal_state = demo[[-1],:]  # shape (1,2)
    # new_trajectry = current_state
    # for _ in range(100):
    # # while np.linalg.norm(goal_state - current_state) > 0.75:
    #     future_state = rk4_step(model, current_state, obstacle_center, radius, dt)
    #     new_trajectry = np.append(new_trajectry, future_state, axis=0)
    #     current_state = future_state
    #     # print(np.linalg.norm(goal_state - current_state))
    # print("Done")
    
    
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    ax.set_aspect(1)
    ax.streamplot(dataXX, dataYY, u, v, density=2)
    ax.scatter(demo[:, 0], demo[:, 1], color=[1, 0, 0])
    # ax.scatter(new_trajectry[:, 0], new_trajectry[:, 1], color=[0.6, 0.35, 0.85])
    ax.plot(new_trajectry[:, 0], new_trajectry[:, 1], lw=3.5, color=[0.6, 0.35, 0.85])
    ax.scatter(surface[:, 0], surface[:, 1], color=[0, 0, 0])
    
    # Add the circle
    circle = Circle((obstacle_center[0,0], obstacle_center[1,0]), radius=radius, edgecolor='black', facecolor='black', linewidth=1)
    ax.add_patch(circle)

#################################################################################################

def modulation_matrix_for_elliptic(state, obstacle_center, r1, r2, m):
    M = np.zeros((state.shape[0],2,2))
    
    q = state.T - obstacle_center  # shape (2, 1000)
    # gradient wrt x,y
    gx = (m/pow(r1,m)) * np.power(q[[0],:], m-1)  # shape (1, 1000)
    gy = (m/pow(r2,m)) * np.power(q[[1],:], m-1)  # shape (1, 1000)
    grad = np.append(gx, gy, axis=0)  # shape (2, 1000)
    
    for i in range(state.shape[0]):
        # Normalize gradient to get normal vector
        n = grad[:,[i]] #/ np.linalg.norm(grad[:,i])  # shape (2, 1)
        
        e = np.cross(np.vstack((n, [0])).reshape(-1), np.array([0,0,1]))  # shape (3,)
        # e = e / np.linalg.norm(e)
        
        E = np.hstack((n, e[0:2].reshape(n.shape)))  # shape (2,2)

        #  Compute distance to the surface
        # d = pow((q[0,i]/r1)**m + (q[1,i]/r2)**m, 1/m)
        d = (q[0,i]/r1)**m + (q[1,i]/r2)**m

        lambda_1 = 1 - 1/(np.abs(d))
        lambda_2 = 1 + 1/(np.abs(d))       
        D = np.diag([lambda_1, lambda_2])
        M[i,:,:] = E @ D @ np.linalg.inv(E)
    return M
    
def plot_modified_vector_field_2(model,datax_grid,datay_grid,demo,surface,obstacle_center, r1, r2, m):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)  #  both will have a shape of (100,100)
    pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))  # (10000,2)

    F = pow(((dataXX-obstacle_center[0,0]) / r1),m) + pow(((dataYY-obstacle_center[1,0]) / r2),m) - 1
    M = modulation_matrix_for_elliptic(pos_array, obstacle_center, r1, r2, m)  # shape (10000,2,2)
    vel = model.predict(pos_array)  # shape (10000,2)
    
    # Modulated velocity
    new_vel = np.squeeze(np.matmul(M, np.expand_dims(vel, axis=2)), axis=2)

    u = new_vel[:, 0].reshape(dataXX.shape)
    v = new_vel[:, 1].reshape(dataXX.shape)

    dt = 1
    goal_state = demo[[-1],:]  # shape (1,2)
    current_state = demo[[0],:]  # shape (1,2)
    new_trajectry = current_state
    for _ in range(demo.shape[0]):
    # while np.linalg.norm(goal_state - current_state) > 1:
        M1 = modulation_matrix_for_elliptic(current_state, obstacle_center, r1, r2, m)  # shape (10000,2,2)
        future_state = current_state.T + (M1[0] @ model.predict(current_state).T) * dt
        new_trajectry = np.append(new_trajectry, future_state.T, axis=0)
        current_state = future_state.T
    
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    ax.set_aspect(1)
    ax.streamplot(dataXX, dataYY, u, v, density=2)
    ax.scatter(demo[:, 0], demo[:, 1], color=[1, 0, 0])
    # ax.scatter(new_trajectry[:, 0], new_trajectry[:, 1], color=[0.6, 0.35, 0.85])
    ax.plot(new_trajectry[:, 0], new_trajectry[:, 1], lw=3.5, color=[0.6, 0.35, 0.85])
    ax.scatter(surface[:, 0], surface[:, 1], color=[0, 0, 0])
    
    # Calculate and plot the contour
    contour_level = 0
    contour = plt.contour(dataXX, dataYY, F, levels=[contour_level])

    # Extract contour paths
    paths = contour.collections[0].get_paths()
    
    # Plot the patch for each path
    for path in paths:
        x_contour, y_contour = path.vertices[:, 0], path.vertices[:, 1]
        plt.fill(x_contour, y_contour, 'k')  # 'b' specifies the color blue

#################################################################################################

def my_modulation_matrix_for_elliptic(state, obstacle_center, r1, r2, m):
    M = np.zeros((state.shape[0],2,2))
    
    q = state.T - obstacle_center  # shape (2, 1000)
    # gradient wrt x,y
    gx = (m/pow(r1,m)) * np.power(q[[0],:], m-1)  # shape (1, 1000)
    gy = (m/pow(r2,m)) * np.power(q[[1],:], m-1)  # shape (1, 1000)
    grad = np.append(gx, gy, axis=0)  # shape (2, 1000)
    
    for i in range(state.shape[0]):
        # Normalize gradient to get normal vector
        n = grad[:,[i]] #/ np.linalg.norm(grad[:,i])  # shape (2, 1)
        
        e = np.cross(np.vstack((n, [0])).reshape(-1), np.array([0,0,1]))  # shape (3,)
        # e = e / np.linalg.norm(e)
        
        E = np.hstack((n, e[0:2].reshape(n.shape)))  # shape (2,2)

        #  Compute distance to the surface
        # d = pow((q[0,i]/r1)**m + (q[1,i]/r2)**m, 1/m)
        d = (q[0,i]/r1)**m + (q[1,i]/r2)**m

        lambda_1 = 1 - 1/(np.abs(d))
        lambda_2 = 1 + 1/(np.abs(d))       
        D = np.diag([lambda_1, lambda_2])
        M[i,:,:] = E @ D @ np.linalg.inv(E)
    return M
    
def plot_modified_vector_field_3(model,datax_grid,datay_grid,demo,surface,obstacle_center, r1, r2, m):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)  #  both will have a shape of (100,100)
    pos_array = np.column_stack((dataXX.ravel(), dataYY.ravel()))  # (10000,2)

    F = pow(((dataXX-obstacle_center[0,0]) / r1),m) + pow(((dataYY-obstacle_center[1,0]) / r2),m) - 1
    M = my_modulation_matrix_for_elliptic(pos_array, obstacle_center, r1, r2, m)  # shape (10000,2,2)
    vel = model.predict(pos_array)  # shape (10000,2)
    
    # Modulated velocity
    new_vel = np.squeeze(np.matmul(M, np.expand_dims(vel, axis=2)), axis=2)

    u = new_vel[:, 0].reshape(dataXX.shape)
    v = new_vel[:, 1].reshape(dataXX.shape)

    dt = 1
    goal_state = demo[[-1],:]  # shape (1,2)
    current_state = demo[[0],:]  # shape (1,2)
    new_trajectry = current_state
    for _ in range(demo.shape[0]):
    # while np.linalg.norm(goal_state - current_state) > 1:
        M1 = my_modulation_matrix_for_elliptic(current_state, obstacle_center, r1, r2, m)  # shape (10000,2,2)
        future_state = current_state.T + (M1[0] @ model.predict(current_state).T) * dt
        new_trajectry = np.append(new_trajectry, future_state.T, axis=0)
        current_state = future_state.T
    
    fig, ax = plt.figure(figsize=(12, 7)), plt.gca()
    ax.set_aspect(1)
    ax.streamplot(dataXX, dataYY, u, v, density=2)
    ax.scatter(demo[:, 0], demo[:, 1], color=[1, 0, 0])
    ax.scatter(new_trajectry[:, 0], new_trajectry[:, 1], color=[0.6, 0.35, 0.85])
    ax.scatter(surface[:, 0], surface[:, 1], color=[0, 0, 0])
    
    # Calculate and plot the contour
    contour_level = 0
    contour = plt.contour(dataXX, dataYY, F, levels=[contour_level])

    # Extract contour paths
    paths = contour.collections[0].get_paths()
    
    # Plot the patch for each path
    for path in paths:
        x_contour, y_contour = path.vertices[:, 0], path.vertices[:, 1]
        plt.fill(x_contour, y_contour, 'k')  # 'b' specifies the color blue

#################################################################################################

def plot_vector_field_minvar(model,datax_grid,datay_grid,demo,surface):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    # orgianize data in an array
    pos_array= np.column_stack((dataXX.ravel(), dataYY.ravel()))
    [vel, std]=model.predict(pos_array, return_std=True)
    grad=model.derivative_of_variance(pos_array).transpose()
    vel_variance_min=vel-2*std*grad/np.linalg.norm(grad, axis=1).reshape(-1,1)
    u= vel_variance_min[:,0].reshape(dataXX.shape)
    v= vel_variance_min[:,1].reshape(dataXX.shape)
    fig = plt.figure(figsize = (12, 7))
    plt.streamplot(dataXX, dataYY, u, v, density = 2)
    plt.scatter(demo[:,0],demo[:,1], color=[1,0,0]) 
    plt.scatter(surface[:,0],surface[:,1], color=[0,0,0])
    plt.title("Minimum variance")

def plot_traj_evolution(model,x_grid,y_grid,z_grid,demo, surface):
    start_pos = np.random.uniform([x_grid[0], y_grid[0], z_grid[0]], [x_grid[-1], y_grid[-1], z_grid[-1]], size=(1, 3))
    traj = np.zeros((1000,3))
    pos=np.array(start_pos).reshape(1,-1)   
    for i in tqdm(range(1000)):
        pos=np.array(pos).reshape(1,-1)

        [vel, std]=model.predict(pos, return_std=True)
        grad=model.derivative_of_variance(pos)
        f_stable=np.array([grad[0,0],grad[1,0],grad[2,0]])/np.sqrt(grad[0,0]**2+grad[1,0]**2+grad[2,0]**2)
        pos = pos+vel.reshape(1,-1)-std[0]*f_stable

        traj[i,:]= pos


    ax = plt.figure().add_subplot(projection='3d')    
    ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)    
    ax.scatter(demo[:,0],demo[:,1],demo[:,2], color=[1,0,0])
    ax.scatter(traj[:,0],traj[:,1],traj[:,2], color=[0,0,1])

def plot_traj_3D(trajectory, surface):

    ax = plt.figure().add_subplot(projection='3d')    
    ax.plot_surface(surface[:,:,0], surface[:,:,1], surface[:,:,2], cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)    
    ax.scatter(trajectory[:,0],trajectory[:,1],trajectory[:,2], color=[0,0,1])

def draw_error_band(ax, x, y, err, loop=False, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err[:,0]
    yp = y + ny * err[:,1]
    xn = x - nx * err[:,0]
    yn = y - ny * err[:,1]

    # print(xp.shape, xn.shape, yp.shape, yn.shape)

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    if loop==True:
        codes[0] = codes[len(xp)] = Path.MOVETO
    codes[0] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, label='Uncertainty',  **kwargs))

def create_vectorfield(model,datax_grid,datay_grid):
    dataXX, dataYY = np.meshgrid(datax_grid, datay_grid)
    pos = np.column_stack((dataXX.ravel(), dataYY.ravel()))
    vel, std = model.predict(pos, return_std=True)
    u, v = vel[:, 0].reshape(dataXX.shape), vel[:, 1].reshape(dataXX.shape)
    return u, v, std
