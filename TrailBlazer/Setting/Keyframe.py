import numpy as np


def get_stt_keyframe(prompt):
    keyframe = []
    left = round(np.random.uniform(low=0.0, high=0.5), 2)
    top = round(np.random.uniform(low=0.0, high=0.5), 2)
    right = round(left + np.random.uniform(low=0.2, high=0.5), 2)
    bottom = round(top + np.random.uniform(low=0.2, high=0.5), 2)
    bbox_ratio = [left, top, right, bottom]
    start = {"bbox_ratios": bbox_ratio, "frame": 0, "prompt": prompt}
    end = {"bbox_ratios": bbox_ratio, "frame": 24, "prompt": prompt}
    keyframe.append(start)
    keyframe.append(end)
    return keyframe


def get_dyn_keyframe(prompt):

    num_keyframes = np.random.randint(5) + 2
    frames = [int(v) for v in np.linspace(0, 24, num_keyframes)]
    bbox_ratios = []
    choice = np.random.randint(6)
    # right to left
    if choice == 0:
        start = [0.0, 0.35, 0.3, 0.65]
        end = [0.7, 0.35, 1.0, 0.65]
    # left to right
    elif choice == 1:
        start = [0.7, 0.35, 1.0, 0.65]
        end = [0.0, 0.35, 0.3, 0.65]
    # top left (small) to bottom right (large)
    elif choice == 2:
        start = [0.0, 0.0, 0.2, 0.2]
        end = [0.5, 0.5, 1.0, 1.0]
    # top left to bottom right
    elif choice == 3:
        start = [0.0, 0.0, 0.5, 0.5]
        end = [0.5, 0.5, 1.0, 1.0]
    # top left (small) to bottom right (large)
    elif choice == 4:
        start = [0.5, 0.5, 1.0, 1.0]
        end = [0.0, 0.0, 0.2, 0.2]
    # top right to bottom left
    elif choice == 5:
        start = [0.5, 0.5, 1.0, 1.0]
        end = [0.0, 0.0, 0.5, 0.5]

    for i in range(num_keyframes):
        if i % 2 == 0:
            bbox_ratios.append(start)
        else:
            bbox_ratios.append(end)

    keyframe = []
    for i in range(num_keyframes):
        keyframe.append(
            {"bbox_ratios": bbox_ratios[i], "frame": frames[i], "prompt": prompt}
        )
    return keyframe
