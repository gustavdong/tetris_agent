from utils import crop_image

# MOVEMENT = [
#     ['NOOP'],
#     ['A'],
#     ['B'],
#     ['right'],
#     ['right', 'A'],
#     ['right', 'B'],
#     ['left'],
#     ['left', 'A'],
#     ['left', 'B'],
#     ['down'],
#     ['down', 'A'],
#     ['down', 'B'],
# ]


# SIMPLE_MOVEMENT = [
#     ['NOOP'],   #0
#     ['A'],      #1
#     ['B'],      #2
#     ['right'],  #3
#     ['left'],   #4
#     ['down'],   #5
# ]

# A|B + 4 x right|left

action_mapping = {  0: [1, 3, 0 ,0 ,0],
                    1: [1, 3, 3, 0, 0],
                    2: [1, 3, 3, 3, 0],
                    3: [1, 3, 3, 3, 3],
                    4: [1, 4, 0, 0 ,0],
                    5: [1, 4, 4, 0, 0],
                    6: [1, 4, 4, 4, 0],
                    7: [1, 4, 4, 4, 4],
                    8: [2, 3, 0 ,0 ,0],
                    9: [2, 3, 3, 0, 0],
                    10: [2, 3, 3, 3, 0],
                    11: [2, 3, 3, 3, 3],
                    12: [2, 4, 0, 0 ,0],
                    13: [2, 4, 4, 0, 0],
                    14: [2, 4, 4, 4, 0],
                    15: [2, 4, 4, 4, 4]
}

def sub_action_loop(action):
    '''
    Loop through preset action sequence to base gym-tetris environment
    once all action sequence finishes, the state will finishes with all '0' 
    sub-actions until tetris piece settles down
    arg: a preset action sequences, <int>
    returns:
        - new state where a new piece of tetris comes out
        - cumulative rewards from action sequence
        - whether the game reaches done state
        - info at the end of action sequence
    '''
    reward = 0
    for i in action_mapping[action]:
        sub_state, sub_reward, done, info = env.action(i)
        reward = max(reward + sub_reward , 0)
        if done:
            return sub_state, reward, done, info
    
    ## mandatory execute 2 down sequence to ensure tetris clears out of top 2 lines of screen
    try: 
        for i in range(2):
            sub_state, sub_reward, done, info = env.action(5)
            reward = max(reward + sub_reward , 0)
            if done:
                return sub_state, reward, done, info
    else: 
        return sub_state, reward, done, info

    ## apply image transformation to substate
    sub_state_crop = crop_image(sub_state)  

    ## finish remaining state with action 'noop'
    while not sub_state_crop[0:2,:][sub_state_crop[0:2,:]>0]:
            sub_state, sub_reward, done, info = env.action(0)
            reward = max(reward + sub_reward , 0)
            sub_state_crop = crop_image(sub_state)

    return sub_state, reward, done, info
