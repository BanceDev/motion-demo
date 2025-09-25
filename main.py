import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection


class AnimationWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create main layout
        layout = QVBoxLayout()

        # Create control button layout
        control_layout = QHBoxLayout()

        # Create play/pause button
        self.play_pause_btn = QPushButton("Pause")
        self.play_pause_btn.clicked.connect(self.toggle_animation)
        self.is_playing = True

        control_layout.addWidget(self.play_pause_btn)
        control_layout.addStretch()  # Add stretch to push button to the left

        # Add control layout to main layout
        layout.addLayout(control_layout)

        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

        # Set up 3D plot
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_zlim([-1.5, 1.5])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        # Scatter points
        self.point1 = self.ax.scatter([], [], [], color="red", s=100, label="Point 1")
        self.point2 = self.ax.scatter([], [], [], color="blue", s=100, label="Point 2")

        # Connecting line
        self.line, = self.ax.plot([], [], [], color="gray", linestyle="--")

        # Trail for point 1
        self.trail_len = 50
        self.trail_points = np.zeros((self.trail_len, 3))
        segments = np.zeros((self.trail_len - 1, 2, 3))
        self.trail = Line3DCollection(segments, cmap="plasma", linewidth=2)
        self.ax.add_collection3d(self.trail)
        self.ax.legend()

        # Animation variables
        self.theta = 0
        self.dtheta = 0.05
        self.frame = 0

        # Set up timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)  # 30ms interval

    def toggle_animation(self):
        """Toggle between play and pause states"""
        if self.is_playing:
            self.timer.stop()
            self.play_pause_btn.setText("Play")
            self.is_playing = False
        else:
            self.timer.start(30)
            self.play_pause_btn.setText("Pause")
            self.is_playing = True

    def update_animation(self):
        # Randomly change the angular speed a bit to simulate acceleration
        self.dtheta += np.random.uniform(-0.005, 0.005)
        self.dtheta = np.clip(self.dtheta, 0.01, 0.15)  # keep speed in reasonable range

        # Update angle
        self.theta += self.dtheta

        # Positions for points
        x1, y1, z1 = np.cos(self.theta), np.sin(self.theta), np.sin(self.theta/2)

        # Blue point moves steadily
        phi = 2 * np.pi * self.frame / 150
        x2, y2, z2 = np.cos(phi), np.sin(phi), np.cos(phi/2)

        # Update scatter points
        self.point1._offsets3d = ([x1], [y1], [z1])
        self.point2._offsets3d = ([x2], [y2], [z2])

        # Update connecting line
        self.line.set_data([x1, x2], [y1, y2])
        self.line.set_3d_properties([z1, z2])

        # Update trail buffer
        self.trail_points = np.roll(self.trail_points, -1, axis=0)
        self.trail_points[-1] = [x1, y1, z1]

        # Create line segments
        segments = np.array([[self.trail_points[i], self.trail_points[i+1]] 
                           for i in range(self.trail_len-1)])
        speeds = np.linalg.norm(np.diff(self.trail_points, axis=0), axis=1)

        # Update trail with color mapping
        self.trail.set_segments(segments)
        self.trail.set_array(speeds)
        self.trail.set_clim(0, 0.2)
        self.trail.set_cmap("hsv")  # purple → red

        # Redraw the canvas
        self.canvas.draw()

        # Increment frame counter
        self.frame += 1


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Animation with PyQt5")
        self.setGeometry(100, 100, 800, 600)

        # Create and set the animation widget
        self.animation_widget = AnimationWidget()
        self.setCentralWidget(self.animation_widget)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
