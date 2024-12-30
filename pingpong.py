import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def process_video(video_path):
    """
    Detect the movement of a ping pong ball in a video using HoughCircles and draw detected circles.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        tuple:
            - X (numpy.ndarray): Array of x-coordinates of the ball.
            - Y (numpy.ndarray): Array of y-coordinates of the ball.
            - T (numpy.ndarray): Array of timestamps corresponding to each frame.
            - R (numpy.ndarray): Array of detected radii of the ball.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate == 0:
        raise ValueError("Could not determine frame rate of the video.")

    X, Y, R, T = [], [], [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray_blurred,
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
            x, y, r = circles[0]

            X.append(x)
            Y.append(y)
            R.append(r)
            T.append(frame_count / frame_rate)

            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

        cv2.imshow('Detected Ball', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    return np.array(X), np.array(Y), np.array(T), np.array(R)

def process_data(x_positions, y_positions, ball_radius):
    """
    Process the ball's positions to shift the origin and calculate conversion factor.

    Parameters:
        x_positions (numpy.ndarray): Array of x-coordinates of the ball.
        y_positions (numpy.ndarray): Array of y-coordinates of the ball.
        ball_radius (numpy.ndarray): Array of radii of the ball in pixels.

    Returns:
        tuple:
            - relative_x_positions (numpy.ndarray): Adjusted x-coordinates.
            - relative_y_positions (numpy.ndarray): Adjusted y-coordinates (upward positive).
            - conversion_factor (float): Conversion factor from pixels to meters.
    """
    initial_x = x_positions[0]
    initial_y = y_positions[0]

    relative_x_positions = x_positions - initial_x
    relative_y_positions = initial_y - y_positions

    radius_pixels = np.mean(ball_radius)
    conversion_factor = 0.02 / radius_pixels  # Assuming the ball radius is 20 mm (0.02 m)

    return relative_x_positions, relative_y_positions, conversion_factor

def plot_ball_route(x_positions, y_positions, x_positions_sim=None, y_positions_sim=None):
    """
    Plot the route of the ping pong ball with a parabolic fit and optional simulated trajectory.

    Parameters:
        x_positions (numpy.ndarray): X coordinates of the ball.
        y_positions (numpy.ndarray): Y coordinates of the ball.
        x_positions_sim (numpy.ndarray, optional): Simulated X coordinates of the ball.
        y_positions_sim (numpy.ndarray, optional): Simulated Y coordinates of the ball.
    """
    coefficients = np.polyfit(x_positions, y_positions, 2)
    polynomial = np.poly1d(coefficients)

    x_fit = np.linspace(np.min(x_positions), np.max(x_positions), len(x_positions))
    y_fit = polynomial(x_fit)

    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, y_positions, label="Ball Route (Measured)", color="blue")
    plt.plot(x_fit, y_fit, label="Parabolic Fit", color="red", linestyle="--")

    if x_positions_sim is not None and y_positions_sim is not None:
        valid_indices = np.where(x_positions_sim <= 1.1)
        plt.plot(x_positions_sim[valid_indices], y_positions_sim[valid_indices], label="Simulated Route (Euler)", color="green", linestyle="--")

    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters, Upward)")
    plt.title("Ping Pong Ball Route with Parabolic Fit and Simulated Trajectory")
    plt.legend()
    plt.grid()
    plt.show()

def euler_method_with_timestamps(v0_x, v0_y, c_divide_m, timestamps):
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

def optimize_parameters(measured_x, measured_y, timestamps, bounds):
    """
    Optimize initial velocities and drag coefficient for the best match of the simulated trajectory.

    Parameters:
        measured_x (numpy.ndarray): Measured x-coordinates.
        measured_y (numpy.ndarray): Measured y-coordinates.
        timestamps (numpy.ndarray): Timestamps for each data point.
        bounds (list of tuple): Bounds for optimization [(v0_x_min, v0_x_max), (v0_y_min, v0_y_max), (c_divide_m_min, c_divide_m_max)].

    Returns:
        tuple: Optimized values for initial velocities and drag coefficient (v0_x, v0_y, c_divide_m).
    """
    def objective(params):
        v0_x, v0_y, c_divide_m = params
        sim_x, sim_y = euler_method_with_timestamps(v0_x, v0_y, c_divide_m, timestamps)
        sim_y_interp = np.interp(measured_x, sim_x, sim_y)
        mse = np.mean((sim_y_interp - measured_y)**2)
        return mse

    initial_guess = np.array([2.2, 2.1, 0.18])
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

    if result.success:
        print(f"Optimized Parameters:\nv0_x: {result.x[0]}\nv0_y: {result.x[1]}\nc/m: {result.x[2]}")
        return result.x
    else:
        raise ValueError("Optimization failed.")

def main(video_path):
    """
    Main function to process the video and plot the ping pong ball's trajectory.

    Parameters:
        video_path (str): Path to the video file.
    """
    x_positions, y_positions, timestamps, ball_radius = process_video(video_path)
    x_positions, y_positions, conversion_factor = process_data(x_positions, y_positions, ball_radius)

    x_positions = x_positions * conversion_factor
    y_positions = y_positions * conversion_factor

    bounds = [(2.0, 4.0), (1.5, 3.5), (0.1, 0.3)]

    # Uncomment the next line to run optimization
    # v0_x, v0_y, c_divide_m = optimize_parameters(x_positions, y_positions, timestamps, bounds)

    v0_x = 2.2827121483095105
    v0_y = 2.078439321009488
    c_divide_m = 0.23170458676029024

    x_positions_sim, y_positions_sim = euler_method_with_timestamps(v0_x, v0_y, c_divide_m, timestamps)

    plot_ball_route(x_positions, y_positions, x_positions_sim, y_positions_sim)

if __name__ == "__main__":
    video_path = 'Ping_pong.mp4'
    main(video_path)
