import cv2
import numpy as np
import matplotlib.pyplot as plt


def process_video(video_path):
    """
    Process the video to track the ping pong ball using Hough Circle Transform.

    Args:
        video_path (str): Path to the video file.

    Returns:
        np.ndarray: x_positions of the ball (NaN for undetected frames).
        np.ndarray: y_positions of the ball (NaN for undetected frames).
        np.ndarray: timestamps corresponding to each frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Cannot open video file.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize arrays for positions and timestamps
    x_positions = np.full(total_frames, np.nan)
    y_positions = np.full(total_frames, np.nan)
    timestamps = np.arange(0, total_frames) / fps

    frame_index = 0
    while True: # Loop through each frame
        ret, frame = cap.read() # Read the frame from the video if ret = false -> no more frames -> break the loop
        if not ret:
            break

        # Preprocess frame: convert to grayscale and apply Gaussian blur for noise reduction
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=30, minRadius=5, maxRadius=50
        )

        # If a circle is detected, store its position
        if circles is not None:
            x, y, _ = circles[0, 0]  # Take the first detected circle
            x_positions[frame_index] = x
            y_positions[frame_index] = y

            #Draw the detected circle
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), 4)

        frame_index += 1


        #Display the frame (press 'q' to quit)
        cv2.imshow("Ping Pong Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()
    return x_positions, y_positions, timestamps


def plot_ball_positions(timestamps, x_positions, y_positions):
    """
    Plot the x and y positions of the ping pong ball over time.

    Args:
        timestamps (np.ndarray): Timestamps for each frame.
        x_positions (np.ndarray): X coordinates of the ball.
        y_positions (np.ndarray): Y coordinates of the ball.
    """
    # Filter out NaN values for continuous plotting
    valid_indices = ~np.isnan(x_positions)
    valid_timestamps = timestamps[valid_indices]
    valid_x_positions = x_positions[valid_indices]
    valid_y_positions = y_positions[valid_indices]

    plt.figure(figsize=(12, 6))

    # X-coordinate vs Time
    plt.subplot(2, 1, 1)
    plt.plot(valid_timestamps, valid_x_positions, label="X Position")
    plt.xlabel("Time (s)")
    plt.ylabel("X Position (pixels)")
    plt.title("Ping Pong Ball X Position Over Time")
    plt.grid()
    plt.legend()

    # Y-coordinate vs Time
    plt.subplot(2, 1, 2)
    plt.plot(valid_timestamps, valid_y_positions, label="Y Position", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Y Position (pixels)")
    plt.title("Ping Pong Ball Y Position Over Time")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def main(video_path):
    """
    Main function to process the video and plot the ping pong ball's trajectory.

    Args:
        video_path (str): Path to the video file.
    """
    x_positions, y_positions, timestamps = process_video(video_path)
    if np.all(np.isnan(x_positions)):
        print("No ping pong ball detected.")
        return
    plot_ball_positions(timestamps, x_positions, y_positions)


# Run the program
if __name__ == "__main__":
    video_path = 'Ping_pong.mp4'
    main(video_path)
