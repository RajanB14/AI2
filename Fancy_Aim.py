from collections import OrderedDict
from pygetwindow import getAllTitles
import win32gui
import torch
import numpy as np
import cv2
import time
import win32api
import win32con
import pandas as pd
from utils.general import non_max_suppression, xyxy2xywh
import dxcam
from scipy.signal import lfilter

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

camera = None  # Define the camera variable in the global scope
alpha = 0.7  # Smoothing factor, you can adjust this value as needed (increase for more smoothing)
max_move = 25  # Maximum mouse movement, adjust as needed
aaQuitKey = "'"  # Define the key to quit the program

# Configuration for frame buffering
frame_buffer_size = 10  # Adjust the buffer size as needed
frame_buffer = []

def add_to_frame_buffer(frame):
    if len(frame_buffer) >= frame_buffer_size:
        frame_buffer.pop(0)  # Remove the oldest frame if the buffer is full
    frame_buffer.append(frame)

def get_average_frame():
    if not frame_buffer:
        return None
    return np.mean(frame_buffer, axis=0).astype(np.uint8)

# Updated prioritize_targets function (same as your original code)
def prioritize_targets(targets, crosshair_x, crosshair_y):
    # Calculate the distance of each target's midpoint to the crosshair
    targets['distance_to_crosshair'] = np.sqrt(
        (targets['current_mid_x'] - crosshair_x) ** 2 + (targets['current_mid_y'] - crosshair_y) ** 2
    )
   
    # You can adjust the weights for size, confidence, and distance as needed
    size_weight = 0.5
    confidence_weight = 0.4
    distance_weight = 0.2
   
    # Calculate a priority score for each target
    targets['priority'] = (
        size_weight * targets['width'] * targets['height'] +
        confidence_weight * targets['confidence'] +
        distance_weight * targets['distance_to_crosshair']
    )
   
    # Sort targets by priority in descending order
    targets = targets.sort_values(by='priority', ascending=False)
   
    return targets

# Placeholder for adjusting the prediction horizon based on target behavior (same as your original code)
def adjust_prediction_horizon(target, prediction_horizon):
    # Adjust the prediction horizon based on the target's velocity
    if target['confidence'] > 0.5:
        prediction_horizon = 0.1  # High confidence target, shorter prediction horizon
    else:
        prediction_horizon = 0.3  # Low confidence target, longer prediction horizon
    return prediction_horizon

# Placeholder for recoil compensation logic (same as your original code)
def compensate_recoil(mouseMove, recoil_amount_x, recoil_amount_y):
    # Compensate for recoil by adjusting mouse movement
    mouseMove[0] -= recoil_amount_x
    mouseMove[1] -= recoil_amount_y
    return mouseMove

# Low-pass filter function (same as your original code)
def low_pass_filter(data, alpha):
    filtered_data = lfilter([alpha], [1, -(1 - alpha)], data, axis=0)
    return filtered_data

def main():
    global camera  # Declare camera as a global variable
    global tracked_objects  # Declare tracked_objects as a global variable

    # Initialize a dictionary to store tracked objects
    tracked_objects = OrderedDict()

    # Configuration and initialization (same as your original code)
    screenShotHeight = 320  # Set to 640 for 640x640 viewable box
    screenShotWidth = 320  # Set to 640 for 640x640 viewable box
    aaRightShift = 0
    aaMovementAmp = .8
    confidence = 0.5
    headshot_mode = True
    cpsDisplay = True
    visuals = True
    prediction_horizon = 0.4  # Define prediction horizon as needed
    recoil_amount_x = 0  # Placeholder for recoil compensation amount in X direction
    recoil_amount_y = 0  # Placeholder for recoil compensation amount in Y direction

    # Load Yolo5 Small AI Model with CUDA support
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True).to(device)
    model.eval()  # Set the model to evaluation mode
    if device.type == 'cuda':
        model.half()  # Use half precision only if CUDA is available

    # Convert the model to float32
    model = model.float()

    # Selecting the game window
    try:
        window_titles = getAllTitles()
        print("=== All Windows ===")
        for index, title in enumerate(window_titles):
            print("[{}]: {}".format(index, title))
        userInput = int(input("Please enter the number corresponding to the window you'd like to select: "))
        selected_window_title = window_titles[userInput]
    except Exception as e:
        print("Failed to select game window: {}".format(e))
        return

    # Activate the game window
    activationRetries = 30
    activationSuccess = False
    while activationRetries > 0:
        try:
            game_window = win32gui.FindWindow(None, selected_window_title)
            if game_window != 0:
                win32gui.SetForegroundWindow(game_window)
                activationSuccess = True
                break
            else:
                print("Failed to find the window with title: {}. Trying again.".format(selected_window_title))
                activationRetries -= 1
                time.sleep(3.0)
        except Exception as e:
            print("Failed to activate game window. Trying again.")
            activationRetries -= 1
            time.sleep(3.0)
    if not activationSuccess:
        return

    # Get screen width and height
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    # Calculate the top-left corner of the screenshot area to center it
    sctArea = {
        "mon": 1,
        "top": (screen_height - screenShotHeight) // 2,
        "left": (screen_width - screenShotWidth) // 2,
        "width": screenShotWidth,
        "height": screenShotHeight
    }

    left = sctArea["left"]
    top = sctArea["top"]
    right, bottom = left + screenShotWidth, top + screenShotHeight
    region = (left, top, right, bottom)

    camera = dxcam.create(region=region)
    if camera is None:
        print("DXCamera failed to initialize.")
        return

    camera.start(target_fps=100, video_mode=True)
    cWidth = sctArea["width"] / 2
    cHeight = sctArea["height"] / 2

    # Force garbage collection
    count = 0
    sTime = time.time()

    COLORS = np.random.uniform(0, 255, size=(1500, 3))

    # Maximum velocity threshold to prevent unnecessary mouse movement
    velocity_threshold = 2  # Experiment with different values

    # Main loop
    last_mid_coord = None
    last_target = None  # Store the last target for velocity calculation
    last_mouse_move = [0, 0]  # Store the last mouse move for smoothing

    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(aaQuitKey)) == 0:
            # Capture the current frame
            npImg = np.array(camera.get_latest_frame())
            
            # Add the current frame to the frame buffer
            add_to_frame_buffer(npImg)
            
            # Get the average frame from the buffer
            average_frame = get_average_frame()
            
            # Image normalization
            im = torch.from_numpy(average_frame).permute(2, 0, 1).float()
            im /= 255
            if len(im.shape) == 3:
                im = im.unsqueeze(0)

            # Get predictions from the YOLO model
            results = model(im)

            # Suppressing results
            pred = non_max_suppression(results, confidence, confidence, 0, False, max_det=1000)

            # Convert output to usable cords
            targets = []
            for i, det in enumerate(pred):
                if len(det):
                    gn = torch.tensor(im.shape)[[0, 0, 0, 0]]
                    for *xyxy, conf, cls in reversed(det):
                        targets.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() + [float(conf)])

            targets = pd.DataFrame(targets, columns=['current_mid_x', 'current_mid_y', 'width', "height", "confidence"])

            # If there are detected people
            if len(targets) > 0:
                if last_mid_coord:
                    targets['last_mid_x'] = last_mid_coord[0]
                    targets['last_mid_y'] = last_mid_coord[1]

                # Prioritize targets based on size, confidence, and distance to crosshair
                targets = prioritize_targets(targets, cWidth, cHeight)

                # Track multiple targets efficiently
                for index, row in targets.iterrows():
                    target = row
                    # Adjust prediction horizon based on target behavior
                    prediction_horizon = adjust_prediction_horizon(target, prediction_horizon)

                    xMid = target.current_mid_x + aaRightShift
                    yMid = target.current_mid_y
                    box_height = target.height
                    headshot_offset = box_height * (0.38 if headshot_mode else 0.2)

                    # Predictive tracking
                    if last_mid_coord and last_target is not None:
                        # Calculate velocity
                        velocity_x = (target.current_mid_x - last_target.current_mid_x) / prediction_horizon
                        velocity_y = (target.current_mid_y - last_target.current_mid_y) / prediction_horizon

                        # Check if velocity exceeds the threshold
                        if abs(velocity_x) > velocity_threshold or abs(velocity_y) > velocity_threshold:
                            xMid = target.current_mid_x + aaRightShift
                            yMid = target.current_mid_y
                            predicted_x = xMid + velocity_x * prediction_horizon
                            predicted_y = yMid + velocity_y * prediction_horizon

                            # Smoothing
                            mouseMove = [predicted_x - cWidth, (predicted_y - headshot_offset) - cHeight]
                            mouseMove = [max(min(move, max_move), -max_move) for move in mouseMove]
                            mouseMove[0] = alpha * mouseMove[0] + (1 - alpha) * last_mouse_move[0]
                            mouseMove[1] = alpha * mouseMove[1] + (1 - alpha) * last_mouse_move[1]
                            last_mouse_move = mouseMove

                            # Apply low-pass filter for enhanced smoothing
                            mouseMove = low_pass_filter([mouseMove], alpha)[0]

                            # Recoil compensation
                            mouseMove = compensate_recoil(mouseMove, recoil_amount_x, recoil_amount_y)

                            # Move the mouse
                            if win32api.GetKeyState(0x14):  # If the CAPS LOCK key is pressed
                                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(mouseMove[0] * aaMovementAmp), int(mouseMove[1] * aaMovementAmp), 0, 0)

                    last_mid_coord = [xMid, yMid]
                    last_target = target

            else:
                last_mid_coord = None
                last_target = None

            if visuals:
                for i in range(len(targets)):
                    halfW = round(targets["width"].iloc[i] / 2)
                    halfH = round(targets["height"].iloc[i] / 2)
                    midX = targets['current_mid_x'].iloc[i]
                    midY = targets['current_mid_y'].iloc[i]
                    (startX, startY, endX, endY) = (int(midX - halfW), int(midY - halfH), int(midX + halfW), int(midY + halfH))
                    label = "Human: {:.2f}%".format(targets["confidence"].iloc[i] * 100)
                    cv2.rectangle(npImg, (startX, startY), (endX, endY), COLORS[0].tolist(), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(npImg, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0].tolist(), 2)

            count += 1
            if time.time() - sTime > 1:
                if cpsDisplay:
                    print("CPS:", count)
                count = 0
                sTime = time.time()

            if visuals:
                npImg = cv2.cvtColor(npImg, cv2.COLOR_BGR2RGB)  # Swap the channels
                cv2.imshow('Live Feed', npImg)
                if cv2.waitKey(1) == ord(';'):
                    break

    camera.stop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Please read the below message and think about how it could be solved before posting it on discord.")
        traceback.print_exception(e)
        print(str(e))
        print("Please read the above message and think about how it could be solved before posting it on discord.")