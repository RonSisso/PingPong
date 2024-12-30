import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def detect_ball_trajectory(video_path):
    """
    Detect the movement of a ping pong ball in a video using HoughCircles and draw detected circles.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        tuple:
            - ball_x_positions (numpy.ndarray): Array of x-coordinates of the ball.
            - ball_y_positions (numpy.ndarray): Array of y-coordinates of the ball.
            - frame_timestamps (numpy.ndarray): Array of timestamps corresponding to each frame.
            - ball_radii (numpy.ndarray): Array of detected radii of the ball.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        raise ValueError("Could not determine frame rate of the video.")

    ball_x_positions, ball_y_positions, ball_radii, frame_timestamps = [], [], [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred_frame,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=30
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, radius = circles[0]

            ball_x_positions.append(x)
            ball_y_positions.append(y)
            ball_radii.append(radius)
            frame_timestamps.append(frame_count / frame_rate)

            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow('Detected Ball', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return np.array(ball_x_positions), np.array(ball_y_positions), np.array(frame_timestamps), np.array(ball_radii)

def preprocess_positions(ball_x_positions, ball_y_positions, ball_radii):
    """
    Preprocess ball positions to shift the origin and calculate conversion factor.

    Parameters:
        ball_x_positions (numpy.ndarray): Array of x-coordinates of the ball.
        ball_y_positions (numpy.ndarray): Array of y-coordinates of the ball.
        ball_radii (numpy.ndarray): Array of radii of the ball in pixels.

    Returns:
        tuple:
            - adjusted_x_positions (numpy.ndarray): Adjusted x-coordinates.
            - adjusted_y_positions (numpy.ndarray): Adjusted y-coordinates (upward positive).
            - pixel_to_meter_factor (float): Conversion factor from pixels to meters.
    """
    initial_x = ball_x_positions[0]
    initial_y = ball_y_positions[0]

    adjusted_x_positions = ball_x_positions - initial_x
    adjusted_y_positions = initial_y - ball_y_positions

    average_radius_pixels = np.mean(ball_radii)
    pixel_to_meter_factor = 0.02 / average_radius_pixels  # Assuming the ball radius is 20 mm (0.02 m)

    return adjusted_x_positions, adjusted_y_positions, pixel_to_meter_factor

def euler_method(v0_x, v0_y, c_divide_m, timestamps):
    """
    Numerically compute the x and y positions of the ball using Euler's method with given timestamps.

    Parameters:
        v0_x (float): Initial velocity in x-direction (m/s).
        v0_y (float): Initial velocity in y-direction (m/s).
        c_divide_m (float): Drag coefficient divided by mass (1/kg).
        timestamps (numpy.ndarray): Array of timestamps (s) for each time step.

    Returns:
        tuple:
            - x_positions (numpy.ndarray): X positions at each time step.
            - y_positions (numpy.ndarray): Y positions at each time step.
    """
    g = 9.8  # Gravitational acceleration (m/s^2)
    num_steps = len(timestamps)

    x_positions = np.zeros(num_steps)
    y_positions = np.zeros(num_steps)
    v_x = np.zeros(num_steps)
    v_y = np.zeros(num_steps)

    x_positions[0] = 0
    y_positions[0] = 0
    v_x[0] = v0_x
    v_y[0] = v0_y

    for i in range(1, num_steps):
        delta_t = timestamps[i] - timestamps[i - 1]
        v_total = np.sqrt(v_x[i-1]**2 + v_y[i-1]**2)

        a_x = -1 * (c_divide_m * v_x[i-1] * v_total)
        a_y = -1 * g - (c_divide_m * v_y[i-1] * v_total)

        v_x[i] = v_x[i-1] + a_x * delta_t
        v_y[i] = v_y[i-1] + a_y * delta_t

        x_positions[i] = x_positions[i-1] + v_x[i-1] * delta_t
        y_positions[i] = y_positions[i-1] + v_y[i-1] * delta_t

        if x_positions[i] > 1.1:
            x_positions = x_positions[:i]
            y_positions = y_positions[:i]
            break

    return x_positions, y_positions

def optimize_trajectory_parameters(measured_x_positions, measured_y_positions, timestamps, parameter_bounds):
    """
    Optimize initial velocities and drag coefficient for the best match of the simulated trajectory.

    Parameters:
        measured_x_positions (numpy.ndarray): Measured x-coordinates.
        measured_y_positions (numpy.ndarray): Measured y-coordinates.
        timestamps (numpy.ndarray): Timestamps for each data point.
        parameter_bounds (list of tuple): Bounds for optimization [(v0_x_min, v0_x_max), (v0_y_min, v0_y_max), (c_divide_m_min, c_divide_m_max)].

    Returns:
        tuple: Optimized values for initial velocities and drag coefficient (v0_x, v0_y, c/m).
    """
    def objective_function(params):
        v0_x, v0_y, c_divide_m = params
        sim_x, sim_y = euler_method(v0_x, v0_y, c_divide_m, timestamps)
        interpolated_sim_y = np.interp(measured_x_positions, sim_x, sim_y)
        mse = np.mean((interpolated_sim_y - measured_y_positions)**2)
        return mse

    initial_guess = np.array([2.2, 2.1, 0.18])
    result = minimize(objective_function, initial_guess, bounds=parameter_bounds, method='L-BFGS-B')

    if result.success:
        print(f"Optimized Parameters:\nInitial Velocity X: {result.x[0]}\nInitial Velocity Y: {result.x[1]}\nDrag Coefficient/Mass: {result.x[2]}")
        return result.x
    else:
        raise ValueError("Optimization failed.")

def plot_ball_trajectory(measured_x_positions, measured_y_positions, simulated_x_positions=None, simulated_y_positions=None):
    """
    Plot the trajectory of the ping pong ball with measured, simulated, and fitted data.

    Parameters:
        measured_x_positions (numpy.ndarray): Measured x positions of the ball.
        measured_y_positions (numpy.ndarray): Measured y positions of the ball.
        simulated_x_positions (numpy.ndarray, optional): Simulated x positions of the ball.
        simulated_y_positions (numpy.ndarray, optional): Simulated y positions of the ball.
    """
    coefficients = np.polyfit(measured_x_positions, measured_y_positions, 2)
    polynomial_fit = np.poly1d(coefficients)

    fitted_x = np.linspace(np.min(measured_x_positions), np.max(measured_x_positions), len(measured_x_positions))
    fitted_y = polynomial_fit(fitted_x)

    plt.figure(figsize=(10, 6))
    plt.plot(measured_x_positions, measured_y_positions, label="Measured Trajectory", color="blue")
    plt.plot(fitted_x, fitted_y, label="Parabolic Fit", color="red", linestyle="--")

    if simulated_x_positions is not None and simulated_y_positions is not None:
        valid_indices = np.where(simulated_x_positions <= 1.1)
        plt.plot(simulated_x_positions[valid_indices], simulated_y_positions[valid_indices], label="Simulated Trajectory", color="green", linestyle="--")

    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Ping Pong Ball Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

def main(video_path):
    """
    Main function to process the video and plot the ping pong ball's trajectory.

    Parameters:
        video_path (str): Path to the video file.
    """
    ball_x_positions, ball_y_positions, frame_timestamps, ball_radii = detect_ball_trajectory(video_path)
    ball_x_positions, ball_y_positions, conversion_factor = preprocess_positions(ball_x_positions, ball_y_positions, ball_radii)

    ball_x_positions = ball_x_positions * conversion_factor
    ball_y_positions = ball_y_positions * conversion_factor

    parameter_bounds = [(2.0, 4.0), (1.5, 3.5), (0.1, 0.3)]

    # Uncomment the next line to run optimization
    # v0_x, v0_y, c_divide_m = optimize_trajectory_parameters(ball_x_positions, ball_y_positions, frame_timestamps, parameter_bounds)

    v0_x = 2.2827121483095105
    v0_y = 2.078439321009488
    c_divide_m = 0.23170458676029024

    simulated_x_positions, simulated_y_positions = euler_method(v0_x, v0_y, c_divide_m, frame_timestamps)

    plot_ball_trajectory(ball_x_positions, ball_y_positions, simulated_x_positions, simulated_y_positions)

if __name__ == "__main__":
    video_path = 'Ping_pong.mp4'
    main(video_path)
