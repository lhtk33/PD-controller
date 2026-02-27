"""
Chapter 2: Simple Oscillator Simulation
========================================

A simplified mass-spring-damper system demonstrating:
- Real-time animation with spring visualization
- Position and velocity plots over time
- Interactive parameter adjustment (mass, damping)

Physical System: m·ẍ + d·ẋ + k·x = 0

Author: Chapter 2 - Modeling Technical Variables
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

# =====================================================
# Simple Oscillator Model
# =====================================================
class SimpleOscillator:
    """
    Mass-spring-damper system (1D oscillation)

    Differential equation: m·ẍ + d·ẋ + k·x = 0

    Technical Variables:
        - position [m]: displacement from equilibrium
        - velocity [m/s]: rate of change of position
        - acceleration [m/s²]: rate of change of velocity
    """

    def __init__(self, mass=0.5, spring_constant=20.0, damping=0.5, initial_position=0.08):
        """Initialize oscillator with parameters.

        Args:
            mass: Mass of oscillating body [kg]
            spring_constant: Spring stiffness [N/m]
            damping: Damping coefficient [Ns/m]
            initial_position: Initial displacement [m]
        """
        # Validate parameters
        assert mass > 0, "Mass must be positive"
        assert spring_constant > 0, "Spring constant must be positive"
        assert damping >= 0, "Damping must be non-negative"

        # System parameters
        self.mass = mass
        self.k = spring_constant
        self.d = damping

        # State variables
        self.position = initial_position    # [m]
        self.velocity = 0.0                 # [m/s]
        self.time = 0.0                     # [s]

        # Store initial conditions
        self.initial_position = initial_position
        self.initial_velocity = 0.0

    def compute_system_info(self):
        """Calculate system parameters for display.

        Returns:
            dict with omega_0, damping_ratio, damping_type
        """
        omega_0 = np.sqrt(self.k / self.mass)  # Natural frequency [rad/s]
        damping_ratio = self.d / (2 * np.sqrt(self.mass * self.k))  # D [-]

        # Classify damping type
        if damping_ratio < 0.01:
            damping_type = "Undamped"
        elif damping_ratio < 1.0:
            damping_type = "Underdamped"
        elif abs(damping_ratio - 1.0) < 0.01:
            damping_type = "Critically Damped"
        else:
            damping_type = "Overdamped"

        return {
            'omega_0': omega_0,
            'damping_ratio': damping_ratio,
            'damping_type': damping_type
        }

    def compute_acceleration(self):
        """Calculate acceleration from current state using Newton's law.

        Force balance: m·a = -k·x - d·v

        Returns:
            Acceleration [m/s²]
        """
        spring_force = -self.k * self.position      # Fs = -k·x [N]
        damping_force = -self.d * self.velocity     # Fd = -d·ẋ [N]

        # Newton's second law: a = F/m
        acceleration = (spring_force + damping_force) / self.mass
        return acceleration

    def step(self, dt):
        """Advance simulation by one timestep using Euler integration.

        Args:
            dt: Time step [s]

        Returns:
            tuple: (position, velocity, acceleration)
        """
        # Compute acceleration at current state
        accel = self.compute_acceleration()

        # Euler integration
        # v(t+dt) = v(t) + a(t)·dt
        self.velocity += accel * dt

        # x(t+dt) = x(t) + v(t+dt)·dt
        self.position += self.velocity * dt

        # Update time
        self.time += dt

        return self.position, self.velocity, accel

    def reset(self):
        """Reset oscillator to initial conditions."""
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        self.time = 0.0


# =====================================================
# Visualization
# =====================================================
class OscillatorVisualizer:
    """Interactive visualization for the oscillator."""

    def __init__(self, oscillator, dt=0.002, total_time=15.0):
        """Initialize visualizer.

        Args:
            oscillator: SimpleOscillator instance
            dt: Simulation timestep [s]
            total_time: Total simulation time [s]
        """
        self.osc = oscillator
        self.dt = dt
        self.total_time = total_time
        self.steps = int(total_time / dt)
        self.time_array = np.linspace(0, total_time, self.steps)

        # History arrays for plotting
        self.position_history = np.zeros(self.steps)
        self.velocity_history = np.zeros(self.steps)
        self.frame = 0

        # Create figure and setup
        self.setup_figure()
        self.create_controls()
        self.update_info_display()

    def setup_figure(self):
        """Create figure with 3 subplots."""
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.suptitle('Simple Oscillator Simulation - Chapter 2',
                         fontsize=14, fontweight='bold')

        # Create grid: 3 rows, 1 column
        # Top: Animation, Middle: Position plot, Bottom: Velocity plot
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1],
                                   left=0.1, right=0.9, top=0.92, bottom=0.25,
                                   hspace=0.4)

        # 1. Animation subplot
        self.ax_anim = self.fig.add_subplot(gs[0])
        self.setup_animation()

        # 2. Position plot
        self.ax_pos = self.fig.add_subplot(gs[1])
        self.setup_position_plot()

        # 3. Velocity plot
        self.ax_vel = self.fig.add_subplot(gs[2])
        self.setup_velocity_plot()

    def setup_animation(self):
        """Setup the animation subplot with spring-mass system."""
        self.ax_anim.set_xlim(-0.15, 0.15)
        self.ax_anim.set_ylim(-0.05, 0.15)
        self.ax_anim.set_title("Mass-Spring-Damper Animation", fontsize=12, fontweight='bold')
        self.ax_anim.set_aspect('equal')
        self.ax_anim.axis('off')

        # Fixed wall
        self.ax_anim.plot([-0.12, -0.12], [0.08, 0.12], 'k-', linewidth=10)
        self.ax_anim.plot([-0.14, -0.10], [0.08, 0.08], 'k-', linewidth=3)

        # Wall hatching
        for i in range(6):
            self.ax_anim.plot([-0.13 + i*0.007, -0.137 + i*0.007],
                            [0.08, 0.065], 'k-', linewidth=1)

        # Spring line (will be animated)
        self.spring_line, = self.ax_anim.plot([], [], 'b-', linewidth=2.5)

        # Mass (blue rectangle)
        self.mass_rect = Rectangle((0, 0), 0.035, 0.035,
                                   facecolor='steelblue',
                                   edgecolor='darkblue',
                                   linewidth=2.5)
        self.ax_anim.add_patch(self.mass_rect)

        # Damper (red line parallel to spring)
        self.damper_line, = self.ax_anim.plot([], [], 'r-', linewidth=2, alpha=0.6)

        # Equilibrium line
        self.ax_anim.axvline(x=0, color='green', linestyle='--',
                            linewidth=1.5, alpha=0.4, label='Equilibrium')

        # Position text
        self.pos_text = self.ax_anim.text(0.0, 0.02, '', fontsize=11,
                                         ha='center',
                                         bbox=dict(boxstyle='round,pad=0.5',
                                                  facecolor='yellow',
                                                  alpha=0.7))

        # Info text (system parameters)
        self.info_text = self.ax_anim.text(-0.13, -0.03, '', fontsize=9,
                                          family='monospace',
                                          verticalalignment='top')

    def draw_spring(self, x_start, x_end, y=0.1, n_coils=10):
        """Generate spring coordinates for visualization.

        Args:
            x_start: Spring start position (wall) [m]
            x_end: Spring end position (mass) [m]
            y: Vertical position of spring [m]
            n_coils: Number of coils to draw

        Returns:
            spring_x, spring_y: Arrays of coordinates
        """
        length = x_end - x_start
        coil_amplitude = 0.012

        # Create spring path
        n_points = n_coils * 4 + 2
        spring_x = np.zeros(n_points)
        spring_y = np.zeros(n_points)

        # Start at wall
        spring_x[0] = x_start
        spring_y[0] = y

        # Create zigzag pattern
        for i in range(1, n_points - 1):
            progress = i / (n_points - 1)
            spring_x[i] = x_start + progress * length

            # Zigzag up and down
            phase = (i - 1) % 4
            if phase == 1:
                spring_y[i] = y + coil_amplitude
            elif phase == 3:
                spring_y[i] = y - coil_amplitude
            else:
                spring_y[i] = y

        # End at mass
        spring_x[-1] = x_end
        spring_y[-1] = y

        return spring_x, spring_y

    def setup_position_plot(self):
        """Setup position vs time plot."""
        self.ax_pos.set_xlim(0, self.total_time)
        self.ax_pos.set_ylim(-0.1, 0.1)
        self.ax_pos.set_xlabel('Time [s]', fontsize=10)
        self.ax_pos.set_ylabel('Position x [m]', fontsize=10)
        self.ax_pos.set_title('Position vs Time', fontsize=11, fontweight='bold')
        self.ax_pos.grid(True, alpha=0.3)

        self.pos_line, = self.ax_pos.plot([], [], 'b-', linewidth=2, label='x(t)')
        self.ax_pos.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_pos.legend(loc='upper right', fontsize=9)

    def setup_velocity_plot(self):
        """Setup velocity vs time plot."""
        self.ax_vel.set_xlim(0, self.total_time)
        self.ax_vel.set_ylim(-1.5, 1.5)
        self.ax_vel.set_xlabel('Time [s]', fontsize=10)
        self.ax_vel.set_ylabel('Velocity v [m/s]', fontsize=10)
        self.ax_vel.set_title('Velocity vs Time', fontsize=11, fontweight='bold')
        self.ax_vel.grid(True, alpha=0.3)

        self.vel_line, = self.ax_vel.plot([], [], 'g-', linewidth=2, label='v(t)')
        self.ax_vel.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_vel.legend(loc='upper right', fontsize=9)

    def create_controls(self):
        """Create interactive sliders and buttons."""
        # Slider for Mass [kg]
        ax_mass = plt.axes([0.2, 0.14, 0.6, 0.02])
        self.slider_mass = Slider(
            ax_mass, 'Mass [kg]',
            valmin=0.1, valmax=2.0,
            valinit=self.osc.mass,
            valstep=0.05
        )

        # Slider for Damping [Ns/m]
        ax_damping = plt.axes([0.2, 0.10, 0.6, 0.02])
        self.slider_damping = Slider(
            ax_damping, 'Damping [Ns/m]',
            valmin=0.0, valmax=5.0,
            valinit=self.osc.d,
            valstep=0.1
        )

        # Slider for Initial Position [m]
        ax_x0 = plt.axes([0.2, 0.06, 0.6, 0.02])
        self.slider_x0 = Slider(
            ax_x0, 'Initial x₀ [m]',
            valmin=0.01, valmax=0.12,
            valinit=self.osc.initial_position,
            valstep=0.01
        )

        # Reset button
        ax_reset = plt.axes([0.42, 0.01, 0.16, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset Simulation',
                               color='lightcoral', hovercolor='salmon')

        # Connect callbacks
        self.slider_mass.on_changed(self.on_parameter_change)
        self.slider_damping.on_changed(self.on_parameter_change)
        self.slider_x0.on_changed(self.on_parameter_change)
        self.btn_reset.on_clicked(self.on_reset)

    def on_parameter_change(self, val):
        """Called when slider values change."""
        self.osc.mass = self.slider_mass.val
        self.osc.d = self.slider_damping.val
        self.osc.initial_position = self.slider_x0.val
        self.update_info_display()

    def update_info_display(self):
        """Update system information text."""
        info = self.osc.compute_system_info()

        info_str = f"""System Parameters:
m = {self.osc.mass:.2f} kg
k = {self.osc.k:.1f} N/m
d = {self.osc.d:.2f} Ns/m

ω₀ = {info['omega_0']:.2f} rad/s
D  = {info['damping_ratio']:.3f}

{info['damping_type']}"""

        self.info_text.set_text(info_str)

    def on_reset(self, event):
        """Reset the simulation."""
        self.osc.reset()
        self.frame = 0
        self.position_history = np.zeros(self.steps)
        self.velocity_history = np.zeros(self.steps)

        # Clear plots
        self.pos_line.set_data([], [])
        self.vel_line.set_data([], [])

    def animate(self, frame_num):
        """Animation update function called each frame."""
        if self.frame >= self.steps:
            return self.get_artists()

        # Simulation step
        x, v, a = self.osc.step(self.dt)

        # Store history
        self.position_history[self.frame] = x
        self.velocity_history[self.frame] = v

        # Update animation
        # Position mass at x (centered)
        mass_x = x - 0.0175
        self.mass_rect.set_xy((mass_x, 0.0825))

        # Update spring
        spring_x, spring_y = self.draw_spring(-0.12, mass_x, y=0.1)
        self.spring_line.set_data(spring_x, spring_y)

        # Update damper (parallel, below spring)
        damper_x, damper_y = self.draw_spring(-0.12, mass_x, y=0.07, n_coils=6)
        self.damper_line.set_data(damper_x, damper_y)

        # Update position text
        self.pos_text.set_text(f'x = {x:.4f} m\nv = {v:.3f} m/s')
        self.pos_text.set_position((x, 0.02))

        # Update plots
        current_time = self.time_array[:self.frame+1]
        self.pos_line.set_data(current_time, self.position_history[:self.frame+1])
        self.vel_line.set_data(current_time, self.velocity_history[:self.frame+1])

        self.frame += 1

        return self.get_artists()

    def get_artists(self):
        """Return all animated artists for blitting."""
        return (self.mass_rect, self.spring_line, self.damper_line,
                self.pos_text, self.pos_line, self.vel_line)

    def run(self):
        """Start the animation."""
        self.ani = FuncAnimation(
            self.fig,
            self.animate,
            frames=self.steps,
            interval=self.dt * 1000,  # Convert to milliseconds
            blit=True,
            repeat=True
        )

        plt.show()


# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SIMPLE OSCILLATOR SIMULATION - Chapter 2")
    print("=" * 70)
    print()
    print("Physical System: m·ẍ + d·ẋ + k·x = 0")
    print()
    print("Technical Variables:")
    print("  • position [m]      - displacement from equilibrium")
    print("  • velocity [m/s]    - rate of change of position")
    print("  • acceleration [m/s²] - rate of change of velocity")
    print()
    print("Controls:")
    print("  • Adjust 'Mass' slider to change system inertia (0.1 - 2.0 kg)")
    print("  • Adjust 'Damping' slider to change energy dissipation (0 - 5 Ns/m)")
    print("  • Adjust 'Initial x₀' to change starting displacement")
    print("  • Click 'Reset Simulation' to restart with current parameters")
    print()
    print("Damping Types:")
    print("  • D ≈ 0:    Undamped (perpetual oscillation)")
    print("  • 0 < D < 1: Underdamped (decaying oscillation)")
    print("  • D ≈ 1:    Critically damped (fastest return, no overshoot)")
    print("  • D > 1:    Overdamped (slow exponential return)")
    print()
    print("=" * 70)

    # Create oscillator with default parameters
    oscillator = SimpleOscillator(
        mass=0.5,               # kg
        spring_constant=20.0,   # N/m (fixed)
        damping=0.5,            # Ns/m (adjustable)
        initial_position=0.08   # m (adjustable)
    )

    # Create and run visualizer
    visualizer = OscillatorVisualizer(
        oscillator,
        dt=0.002,      # 2 ms timestep (500 Hz)
        total_time=15.0  # 15 seconds
    )

    # Display system info
    info = oscillator.compute_system_info()
    print(f"\nInitial Configuration:")
    print(f"  Natural Frequency ω₀ = {info['omega_0']:.3f} rad/s")
    print(f"  Damping Ratio D = {info['damping_ratio']:.3f}")
    print(f"  Damping Type: {info['damping_type']}")
    print()
    print("Starting simulation...")
    print("=" * 70)

    visualizer.run()