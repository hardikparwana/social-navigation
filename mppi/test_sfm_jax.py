import socialforce
import jax.numpy as jnp
import matplotlib.pyplot as plt



if 1:
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(xlim=(-1,10),ylim=(-1,10))
    ax.set_xlabel("X"); ax.set_ylabel("Y")

    # x, y, vx, vy, goal_x, goal_y
    initial_state = jnp.array([
        [0.0, 0.0, 0.5, 0.5, 10.0, 10.0], # 1
        [10.0, 0.3, -0.5, 0.5, 0.0, 10.0], # 2
    ])
    humans = ax.scatter(initial_state[:,0], initial_state[:,1],c='g',alpha=0.5,s=50)

    t = 0
    dt = 0.05
    tf = 5.0
    # social_state = initial_state #jnp.append( initial_state, jnp.array([ [0], [0] ]), axis=1 )
    
    tau = 0.5
    tau = tau * jnp.ones(initial_state.shape[0])
    social_state = jnp.concatenate((initial_state, jnp.expand_dims(tau, -1)), axis=-1)
    s = socialforce.Simulator( social_state, delta_t = dt )
    state = social_state

    MAX_SPEED_MULTIPLIER = 1.3
    initial_speed = 1.0
    initial_speeds = jnp.ones((initial_state.shape[0])) * initial_speed
    max_speeds = MAX_SPEED_MULTIPLIER * initial_speeds

    while (t<tf):
        F, state = s.step(state, initial_speeds, max_speeds, dt)
        humans.set_offsets(state[:,0:2])

        fig.canvas.draw()
        fig.canvas.flush_events()