import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
import math

class QuadcopterCKFPIControl:
    def __init__(self):
        # System parameters
        self.m = 1.5  # Drone mass (kg)
        self.Ix = 0.02  # Angular momentum (roll) (kg*m^2)
        self.Iy = 0.02  # Angular momentum (pitch) (kg*m^2)
        self.Iz = 0.04  # Angular momentum (yaw) (kg*m^2)
        self.l = 0.1  # Arm length (m)
        self.g = 9.81  # Gravity (m/s^2)
        
        # Motor parameters
        self.R = 0.1  # Winding resistance (Ohm)
        self.L = 0.002  # Winding inductance (H)
        self.kb = 0.01  # Back EMF constant (Vs/rad)
        self.kt = 0.01  # Torque constant (N/A)
        self.bm = 0.00009  # Motor damping coefficient (Ns/rad)
        self.Jm = 0.0004  # Motor/propeller inertia (kg*m^2)
        self.k_drag = 0.000002  # Propeller aerodynamic drag coefficient
        self.k_thrust = 0.000018  # Propeller thrust coefficient
        
        # Damping coefficients
        self.kdx = 0.15  # Velocity damping (x-axis) (Ns/m)
        self.kdy = 0.15  # Velocity damping (y-axis) (Ns/m)
        self.kdz = 0.25  # Velocity damping (z-axis) (Ns/m)
        self.k_phi = 0.002  # Viscous damping (roll) (Ns/rad)
        self.k_theta = 0.002  # Viscous damping (pitch) (Ns/rad)
        self.k_psi = 0.003  # Viscous damping (yaw) (Ns/rad)
        
        # Sampling time
        self.Ts = 0.004  # Sampling time (s)
        
        # Calculate hover speed
        self.w_hover = np.sqrt(self.m * self.g / (4 * self.k_thrust))
        print(f"Hover speed: {self.w_hover:.2f} rad/s")
        
        # PI Controller gains (to be tuned)
        # Attitude controllers
        self.Kp_roll = 15.0
        self.Ki_roll = 5.0
        self.Kp_pitch = 15.0
        self.Ki_pitch = 5.0
        self.Kp_yaw = 5.0
        self.Ki_yaw = 1.0
        
        # Altitude controller
        self.Kp_alt = 8.0
        self.Ki_alt = 1.0
        
        # Velocity controller
        self.K_vel = 0.5
        
        # CKF parameters
        self.Q = np.eye(20) * 0.01  # Process noise covariance
        self.R_ckf = np.diag([0.01, 0.01, 0.01, 0.01])  # Measurement noise covariance
        self.P = np.eye(20) * 0.1  # Initial state covariance
        
        # Initialize state
        self.reset_state()
    
    def reset_state(self):
        """Initialize the state vector"""
        # State vector: [i1, i2, i3, i4, w1, w2, w3, w4, x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.zeros(20)
        
        # Initialize motor speeds to hover speed
        self.state[4:8] = self.w_hover  # Motor speeds w1, w2, w3, w4
        
        # Integral terms for PI controllers
        self.integral_roll = 0.0
        self.integral_pitch = 0.0
        self.integral_yaw = 0.0
        self.integral_alt = 0.0
    
    def motor_dynamics(self, i_motor, w_motor, v_applied):
        """Calculate motor electrical and mechanical dynamics"""
        di_dt = (1/self.L) * (-self.R * i_motor - self.kb * w_motor + v_applied)
        dw_dt = (1/self.Jm) * (self.kt * i_motor - self.bm * w_motor - self.k_drag * w_motor**2)
        return di_dt, dw_dt
    
    def translational_dynamics(self, state):
        """Calculate translational dynamics"""
        # Extract states
        phi, theta, psi = state[14], state[15], state[16]
        vx, vy, vz = state[11], state[12], state[13]
        
        # Calculate total thrust from all motors
        w_motors = state[4:8]
        T_total = np.sum(self.k_thrust * w_motors**2)
        
        # Translational accelerations
        dvx_dt = (T_total / self.m) * (np.cos(phi)*np.sin(theta)*np.cos(psi) + np.sin(phi)*np.sin(psi)) - (self.kdx / self.m) * vx
        dvy_dt = (T_total / self.m) * (np.cos(phi)*np.sin(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)) - (self.kdy / self.m) * vy
        dvz_dt = (T_total / self.m) * (np.cos(phi)*np.cos(theta)) - self.g - (self.kdz / self.m) * vz
        
        return dvx_dt, dvy_dt, dvz_dt
    
    def rotational_dynamics(self, state, tau_x, tau_y, tau_z):
        """Calculate rotational dynamics"""
        # Extract angular velocities
        p, q, r = state[17], state[18], state[19]
        
        # Angular accelerations (body frame)
        dp_dt = (1/self.Ix) * (tau_x + (self.Iy - self.Iz) * q * r) - (self.k_phi / self.Ix) * p
        dq_dt = (1/self.Iy) * (tau_y + (self.Iz - self.Ix) * p * r) - (self.k_theta / self.Iy) * q
        dr_dt = (1/self.Iz) * (tau_z + (self.Ix - self.Iy) * p * q) - (self.k_psi / self.Iz) * r
        
        return dp_dt, dq_dt, dr_dt
    
    def euler_kinematics(self, state):
        """Calculate Euler angle kinematics"""
        phi, theta, psi = state[14], state[15], state[16]
        p, q, r = state[17], state[18], state[19]
        
        # Euler angle derivatives
        dphi_dt = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
        dtheta_dt = q * np.cos(phi) - r * np.sin(phi)
        dpsi_dt = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)
        
        return dphi_dt, dtheta_dt, dpsi_dt
    
    def system_dynamics(self, state, u_voltages):
        """Full system dynamics"""
        # Motor dynamics (for each motor)
        di_dt = np.zeros(4)
        dw_dt = np.zeros(4)
        for i in range(4):
            di_dt[i], dw_dt[i] = self.motor_dynamics(state[i], state[i+4], u_voltages[i])
        
        # Translational dynamics
        dvx_dt, dvy_dt, dvz_dt = self.translational_dynamics(state)
        
        # Calculate torques from motor speeds (X-configuration)
        w_sqr = state[4:8]**2
        tau_x = (self.l / np.sqrt(2)) * (self.k_thrust * (w_sqr[1] + w_sqr[2] - w_sqr[0] - w_sqr[3]))
        tau_y = (self.l / np.sqrt(2)) * (self.k_thrust * (w_sqr[0] + w_sqr[1] - w_sqr[2] - w_sqr[3]))
        tau_z = self.k_drag * (w_sqr[0] - w_sqr[1] + w_sqr[2] - w_sqr[3])
        
        # Limit torques to prevent instability
        max_tau = 2.0  # Maximum torque (N*m)
        tau_x = np.clip(tau_x, -max_tau, max_tau)
        tau_y = np.clip(tau_y, -max_tau, max_tau)
        tau_z = np.clip(tau_z, -max_tau, max_tau)
        
        # Rotational dynamics
        dp_dt, dq_dt, dr_dt = self.rotational_dynamics(state, tau_x, tau_y, tau_z)
        
        # Euler kinematics
        dphi_dt, dtheta_dt, dpsi_dt = self.euler_kinematics(state)
        
        # Position derivatives
        dx_dt = state[11]  # vx
        dy_dt = state[12]  # vy
        dz_dt = state[13]  # vz
        
        # Pack all derivatives
        derivatives = np.concatenate([
            di_dt,          # Motor currents (4)
            dw_dt,          # Motor speeds (4)
            [dx_dt, dy_dt, dz_dt],  # Position (3)
            [dvx_dt, dvy_dt, dvz_dt],  # Velocities (3)
            [dphi_dt, dtheta_dt, dpsi_dt],  # Euler angles (3)
            [dp_dt, dq_dt, dr_dt]   # Angular velocities (3)
        ])
        
        return derivatives
    
    def forward_euler_step(self, state, u_voltages, dt):
        """Forward Euler integration step"""
        derivatives = self.system_dynamics(state, u_voltages)
        new_state = state + dt * derivatives
        return new_state
    
    def ckf_predict(self, state, P, Q):
        """CKF prediction step"""
        n = len(state)
        # Generate cubature points
        sqrt_n = np.sqrt(n)
        chol_P = cholesky(P, lower=True)
        
        # Points are generated as state + sqrt_n * chol_P * xi where xi are unit vectors
        points = np.tile(state[:, np.newaxis], (1, 2*n))
        
        for i in range(n):
            points[:, i] = state + sqrt_n * chol_P[:, i]
            points[:, i+n] = state - sqrt_n * chol_P[:, i]
        
        # Propagate points through dynamics (simplified for this example)
        # We'll use a simple propagation model here
        dt = self.Ts
        propagated_points = np.zeros_like(points)
        for j in range(2*n):
            # Use actual control inputs for propagation
            # For now, we will use the same control for all points as an approximation
            # In a real implementation, we would have the control based on each state
            u_actual = np.ones(4) * 12.0  # Placeholder - in reality this would depend on the specific state
            u_hover = np.ones(4) * 12.0  # Nominal voltage
            propagated_points[:, j] = self.forward_euler_step(points[:, j], u_hover, dt)
        
        # Compute predicted mean
        x_pred = np.mean(propagated_points, axis=1)
        # Ensure P_pred remains positive definite and symmetric
        # Ensure P_pred remains positive definite and symmetric
        P_pred = (P_pred + P_pred.T) / 2.0
        eigenvals = np.linalg.eigvals(P_pred)
        if np.any(eigenvals <= 0):
            P_pred += np.eye(n) * 0.001  # Add small positive value to diagonal
        P_pred = (P_pred + P_pred.T) / 2.0
        eigenvals = np.linalg.eigvals(P_pred)
        if np.any(eigenvals <= 0):
        # Compute predicted covariance
        # Ensure P_pred remains positive definite and symmetric
        P_pred = (P_pred + P_pred.T) / 2.0
        eigenvals = np.linalg.eigvals(P_pred)
        if np.any(eigenvals <= 0):
            P_pred += np.eye(n) * 0.001  # Add small positive value to diagonal
        diff = propagated_points - x_pred[:, np.newaxis]
        P_pred = diff @ diff.T / (2*n) + Q
        
        return x_pred, P_pred
    
    def ckf_update(self, x_pred, P_pred, measurement, R):
        """CKF update step"""
        # Measurement matrix (simplified) - measuring z, phi, theta, psi
        H = np.zeros((4, len(x_pred)))
        H[0, 10] = 1  # Measure z (height)
        H[1, 14] = 1  # Measure phi (roll)
        H[2, 15] = 1  # Measure theta (pitch)
        H[3, 16] = 1  # Measure psi (yaw)
        
        # Predicted measurement
        y_pred = H @ x_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = measurement - y_pred
        
        # Update state and covariance
        x_updated = x_pred + K @ innovation
        P_updated = P_pred - K @ S @ K.T
        
        return x_updated, P_updated
    
    def attitude_pi_control(self, phi_ref, theta_ref, psi_ref, phi_est, theta_est, psi_est):
        """Attitude PI controller"""
        # Errors
        e_roll = phi_ref - phi_est
        e_pitch = theta_ref - theta_est
        e_yaw = psi_ref - psi_est
        
        # Update integral terms
        self.integral_roll += e_roll * self.Ts
        self.integral_pitch += e_pitch * self.Ts
        self.integral_yaw += e_yaw * self.Ts
        
        # Apply integral windup protection
        max_integral = 2.0
        self.integral_roll = np.clip(self.integral_roll, -max_integral, max_integral)
        self.integral_pitch = np.clip(self.integral_pitch, -max_integral, max_integral)
        self.integral_yaw = np.clip(self.integral_yaw, -max_integral, max_integral)
        
        # PI control outputs
        u_roll = self.Kp_roll * e_roll + self.Ki_roll * self.integral_roll
        u_pitch = self.Kp_pitch * e_pitch + self.Ki_pitch * self.integral_pitch
        u_yaw = self.Kp_yaw * e_yaw + self.Ki_yaw * self.integral_yaw
        
        return u_roll, u_pitch, u_yaw
    
    def altitude_pi_control(self, z_ref, z_est):
        """Altitude PI controller"""
        e_alt = z_ref - z_est
        
        # Update integral term
        self.integral_alt += e_alt * self.Ts
        
        # Apply integral windup protection
        max_integral_alt = 8.0
        self.integral_alt = np.clip(self.integral_alt, -max_integral_alt, max_integral_alt)
        
        # PI control output
        u_alt = self.Kp_alt * e_alt + self.Ki_alt * self.integral_alt
        
        return u_alt
    
    def motor_mixing(self, thrust_cmd, tau_x, tau_y, tau_z):
        """Convert desired thrust and torques to motor voltages (X-configuration)"""
        # For X-configuration, mixing matrix
        # [T]     [1   0   0   0] [V1]
        # [tx] =  [0  l/√2  0  -kq] [V2]
        # [ty]    [0  l/√2 -kq  0] [V3]
        # [tz]    [0   0   kq  -kq] [V4]

        # Calculate required motor forces based on thrust and torques
        # For X-configuration quadcopter
        # The control allocation matrix maps desired forces/torques to motor commands

        # Normalize control efforts to prevent actuator saturation
        max_thrust_total = 4.0 * self.m * self.g  # 4x hover thrust total
        max_tau_x = max_thrust_total * self.l / 2  # Maximum possible torque around x
        max_tau_y = max_thrust_total * self.l / 2  # Maximum possible torque around y
        max_tau_z = max_thrust_total * 0.1  # Maximum possible torque around z (based on drag)

        tau_x_clipped = np.clip(tau_x, -max_tau_x, max_tau_x)
        tau_y_clipped = np.clip(tau_y, -max_tau_y, max_tau_y)
        tau_z_clipped = np.clip(tau_z, -max_tau_z, max_tau_z)

        F_desired = np.array([
            thrust_cmd/4 - tau_x_clipped/(2*np.sqrt(2)*self.l) - tau_y_clipped/(2*self.l) + tau_z_clipped/4,
            thrust_cmd/4 + tau_x_clipped/(2*np.sqrt(2)*self.l) - tau_y_clipped/(2*self.l) - tau_z_clipped/4,
            thrust_cmd/4 + tau_x_clipped/(2*np.sqrt(2)*self.l) + tau_y_clipped/(2*self.l) + tau_z_clipped/4,
            thrust_cmd/4 - tau_x_clipped/(2*np.sqrt(2)*self.l) + tau_y_clipped/(2*self.l) - tau_z_clipped/4
        ])

        # Ensure all motor forces are positive and within safe bounds
        F_min = 0.05 * self.m * self.g / 4  # Minimum 5% hover force per motor
        F_max = 1.5 * self.m * self.g / 4   # Maximum 150% hover force per motor
        F_desired = np.clip(F_desired, F_min, F_max)

        # Convert forces to required motor speeds
        w_required = np.sqrt(F_desired / self.k_thrust)
        w_required = np.clip(w_required, 0, 600)  # Max speed limit

        # Calculate required voltages to achieve desired speeds
        voltages = np.zeros(4)
        for i in range(4):
            # At steady state: V = R*i + kb*w, and kt*i = bm*w + k_drag*w^2
            # So i = (bm*w + k_drag*w^2)/kt
            # And V = R*(bm*w + k_drag*w^2)/kt + kb*w
            req_current = (self.bm * w_required[i] + self.k_drag * w_required[i]**2) / self.kt
            voltages[i] = self.R * req_current + self.kb * w_required[i]

        # Apply voltage limits
        voltages = np.clip(voltages, 0, 12.0)

        return voltages
        voltages = np.clip(voltages, 0, 12.0)
        
        return voltages
    
    def simulate_attitude_control(self, duration=10.0):
        """Simulate attitude control with CKF"""
        num_steps = int(duration / self.Ts)
        
        # Store results
        t = np.linspace(0, duration, num_steps)
        motor_voltages = np.zeros((num_steps, 4))
        motor_currents = np.zeros((num_steps, 4))
        motor_speeds = np.zeros((num_steps, 4))
        height = np.zeros(num_steps)
        roll = np.zeros(num_steps)
        pitch = np.zeros(num_steps)
        yaw = np.zeros(num_steps)
        
        # Reference signals
        z_ref = 2.0  # Target height
        phi_ref = 0.0  # Target roll
        theta_ref = 0.0  # Target pitch
        psi_ref = 0.0  # Target yaw
        
        # Reset initial conditions
        self.reset_state()
        
        # Add small initial disturbance
        self.state[14] = 0.1  # Initial roll angle
        self.state[15] = 0.05  # Initial pitch angle
        
        print("Starting attitude control simulation...")
        
        for i in range(num_steps):
            # CKF estimation
            # Create measurement (with some noise)
            measurement = np.array([
                self.state[10],  # z (height)
                self.state[14],  # phi (roll)
                self.state[15],  # theta (pitch)
                self.state[16]   # psi (yaw)
            ]) + np.random.normal(0, 0.001, 4)  # Add measurement noise
            
            # CKF predict and update
            self.state, self.P = self.ckf_predict(self.state, self.P, self.Q)
            self.state, self.P = self.ckf_update(self.state, self.P, measurement, self.R_ckf)
            
            # Extract estimated states for control
            z_est = self.state[10]
            phi_est = self.state[14]
            theta_est = self.state[15]
            psi_est = self.state[16]
            
            # PI Controllers
            # Altitude control
            u_alt = self.altitude_pi_control(z_ref, z_est)
            
            # Calculate required thrust
            T_req = self.m * self.g + u_alt
            
            # Attitude control
            u_tau_x, u_tau_y, u_tau_z = self.attitude_pi_control(
                phi_ref, theta_ref, psi_ref, 
                phi_est, theta_est, psi_est
            )
            
            # Motor mixing
            voltages = self.motor_mixing(T_req, u_tau_x, u_tau_y, u_tau_z)
            
            # Forward Euler step
            self.state = self.forward_euler_step(self.state, voltages, self.Ts)
            
            # Store results
            motor_voltages[i, :] = voltages
            motor_currents[i, :] = self.state[0:4]
            motor_speeds[i, :] = self.state[4:8]
            height[i] = self.state[10]
            roll[i] = self.state[14]
            pitch[i] = self.state[15]
            yaw[i] = self.state[16]
            
            # Print progress
            if i % (num_steps // 10) == 0:
                print(f"Progress: {100*i/num_steps:.1f}% - Height: {height[i]:.2f}, Roll: {roll[i]:.3f}")
        
        print("Simulation completed!")
        
        # Plot results
        self.plot_results(t, motor_voltages, motor_currents, motor_speeds, height, roll, pitch, yaw, z_ref)
        
        return t, motor_voltages, motor_currents, motor_speeds, height, roll, pitch, yaw
    
    def plot_results(self, t, motor_voltages, motor_currents, motor_speeds, height, roll, pitch, yaw, z_ref):
        """Plot simulation results"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        
        # Motor voltages
        axes[0, 0].plot(t, motor_voltages[:, 0], label='M1', linewidth=1)
        axes[0, 0].plot(t, motor_voltages[:, 1], label='M2', linewidth=1)
        axes[0, 0].plot(t, motor_voltages[:, 2], label='M3', linewidth=1)
        axes[0, 0].plot(t, motor_voltages[:, 3], label='M4', linewidth=1)
        axes[0, 0].set_title('Motor Voltages (V)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Voltage (V)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Motor currents
        axes[0, 1].plot(t, motor_currents[:, 0], label='M1', linewidth=1)
        axes[0, 1].plot(t, motor_currents[:, 1], label='M2', linewidth=1)
        axes[0, 1].plot(t, motor_currents[:, 2], label='M3', linewidth=1)
        axes[0, 1].plot(t, motor_currents[:, 3], label='M4', linewidth=1)
        axes[0, 1].set_title('Motor Currents (A)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Current (A)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Motor speeds
        axes[1, 0].plot(t, motor_speeds[:, 0], label='M1', linewidth=1)
        axes[1, 0].plot(t, motor_speeds[:, 1], label='M2', linewidth=1)
        axes[1, 0].plot(t, motor_speeds[:, 2], label='M3', linewidth=1)
        axes[1, 0].plot(t, motor_speeds[:, 3], label='M4', linewidth=1)
        axes[1, 0].axhline(y=self.w_hover, color='r', linestyle='--', label='Hover Speed')
        axes[1, 0].set_title('Propeller Speeds (rad/s)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Speed (rad/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Height
        axes[1, 1].plot(t, height, linewidth=2, label='Actual')
        axes[1, 1].axhline(y=z_ref, color='r', linestyle='--', label='Reference')
        axes[1, 1].set_title('Height Response (m)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Height (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Roll
        axes[2, 0].plot(t, np.degrees(roll), linewidth=2)
        axes[2, 0].set_title('Roll Angle Response (deg)')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Roll (deg)')
        axes[2, 0].grid(True)
        
        # Pitch
        axes[2, 1].plot(t, np.degrees(pitch), linewidth=2)
        axes[2, 1].set_title('Pitch Angle Response (deg)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Pitch (deg)')
        axes[2, 1].grid(True)
        
        # Yaw
        axes[3, 0].plot(t, np.degrees(yaw), linewidth=2)
        axes[3, 0].set_title('Yaw Angle Response (deg)')
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('Yaw (deg)')
        axes[3, 0].grid(True)
        
        # Empty subplot
        axes[3, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the simulation"""
    print("Initializing Quadcopter CKF-PI Control Simulation...")
    
    # Create controller instance
    controller = QuadcopterCKFPIControl()
    
    # Run simulation
    try:
        t, motor_voltages, motor_currents, motor_speeds, height, roll, pitch, yaw = controller.simulate_attitude_control(duration=10.0)
        
        # Calculate performance metrics
        print("\nPerformance Metrics:")
        print(f"Final height: {height[-1]:.3f} m (ref: 2.0 m)")
        print(f"Final roll: {np.degrees(roll[-1]):.3f} deg")
        print(f"Final pitch: {np.degrees(pitch[-1]):.3f} deg")
        print(f"Final yaw: {np.degrees(yaw[-1]):.3f} deg")
        
        # Check settling time and overshoot for height
        target_height = 2.0
        initial_height = height[0]
        tolerance = 0.02 * target_height  # 2% of target
        
        # Find settling time
        for i in range(len(height)-1, 0, -1):
            if abs(height[i] - target_height) > tolerance:
                settling_idx = i
                break
        else:
            settling_idx = 0
        
        settling_time = t[settling_idx] if settling_idx < len(t) else t[-1]
        print(f"Settling time: {settling_time:.3f} s")
        
        # Overshoot
        max_height = np.max(height[int(0.1*len(height)):])  # Exclude initial transient
        overshoot_pct = max((max_height - target_height) / target_height * 100, 0)
        print(f"Overshoot: {overshoot_pct:.2f}%")
        
        # Check if requirements are met
        print(f"\nRequirements Check:")
        print(f"Height overshoot < 10%: {'PASS' if overshoot_pct < 10 else 'FAIL'} ({overshoot_pct:.2f}%)")
        print(f"Settling time < 3s: {'PASS' if settling_time < 3.0 else 'FAIL'} ({settling_time:.3f}s)")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()