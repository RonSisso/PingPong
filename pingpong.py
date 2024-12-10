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
    ball_radius = np.full(total_frames, np.nan)
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
            x, y, radius = circles[0, 0]  # Take the first detected circle
            x_positions[frame_index] = x
            y_positions[frame_index] = y
            ball_radius[frame_index] = radius

            #Draw the detected circle
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), 3)

        frame_index += 1


        #Display the frame (press 'q' to quit)
        cv2.imshow("Ping Pong Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()
    return x_positions, y_positions, timestamps, ball_radius

def process_data(x_positions, y_positions, timestamps):

    # Filter out NaN values for continuous plotting
    valid_indices = ~np.isnan(x_positions)
    valid_x_positions = x_positions[valid_indices]
    valid_y_positions = y_positions[valid_indices]
    timestamps = timestamps[valid_indices]

    # Get the initial position of the ball
    initial_x = valid_x_positions[0]
    initial_y = valid_y_positions[0]

    # Shift positions to make the first frame the origin
    relative_x_positions = valid_x_positions - initial_x
    relative_y_positions = initial_y - valid_y_positions  # Inverted to make upward movement positive

    return relative_x_positions, relative_y_positions, timestamps


def calculate_velocities(x_positions, y_positions, conversion_factor):
    """
    Calculate the velocity (m/s) using numpy.gradient.

    Args:
        x_positions (np.ndarray): X positions of the object in pixels.
        y_positions (np.ndarray): Y positions of the object in pixels.
        conversion_factor (float): Conversion factor from pixels to meters.

    Returns:
        np.ndarray: velocities (m/s) for each frame.
    """

    dt = 1/2000
    # Convert positions to meters
    x_positions_meters = x_positions * conversion_factor
    y_positions_meters = y_positions * conversion_factor

    # Compute gradients (velocities) in meters per second
    v_x = np.gradient(x_positions_meters, dt)  # Velocity in x-direction
    v_y = np.gradient(y_positions_meters, dt)  # Velocity in y-direction

    # Compute total velocity
    v = np.sqrt(v_x**2 + v_y**2)

    return v



def plot_ball_route(x_positions, y_positions):
    """
    Plot the route of the ping pong ball with a parabolic fit.

    Args:
        x_positions (np.ndarray): X coordinates of the ball.
        y_positions (np.ndarray): Y coordinates of the ball.
    """
    # Perform a parabolic fit to the ball route
    coefficients = np.polyfit(x_positions, y_positions, 2)  # Fit y = ax^2 + bx + c
    polynomial = np.poly1d(coefficients)

    # Generate a smooth curve for the fitted parabola
    x_fit = np.linspace(np.min(x_positions), np.max(x_positions), len(x_positions))
    y_fit = polynomial(x_fit)

    # Plot the original ball route
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, y_positions, label="Ball Route", color="blue")

    # Plot the fitted parabolic curve
    plt.plot(x_fit, y_fit, label="Parabolic Fit", color="red", linestyle="--")

    # Labels and title
    plt.xlabel("X Position (pixels)")
    plt.ylabel("Y Position (pixels, Upward)")
    plt.title("Ping Pong Ball Route with Parabolic Fit")
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()




def main(video_path):
    """
    Main function to process the video and plot the ping pong ball's trajectory.

    Args:
        video_path (str): Path to the video file.
    """

    x_positions, y_positions, timestamps, ball_radius = process_video(video_path)

    if np.all(np.isnan(x_positions)):
        print("No ping pong ball detected.")
        return

    x_positions, y_positions, timestamps = process_data(x_positions, y_positions, timestamps)

    # Calculate the conversion factor
    valid_radii = ball_radius[~np.isnan(ball_radius)]
    radius_pixels = np.mean(valid_radii)
    conversion_factor = 0.02 / radius_pixels  # Ball radius is 20 mm (0.02 m)

    plot_ball_route(x_positions, y_positions)

    # Calculate total velocity for ball and racket
    v_total_ball = calculate_velocities(x_positions, y_positions, conversion_factor)
    print(v_total_ball)




# Run the program
if __name__ == "__main__":
    video_path = 'Ping_pong.mp4'
    main(video_path)
