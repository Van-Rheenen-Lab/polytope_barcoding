import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QSlider,
    QLabel,
    QFileDialog,
    QCheckBox,
)
from skimage.measure import find_contours
from skimage.transform import downscale_local_mean
import numpy as np
from skimage.filters import threshold_otsu
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class Annotator(QMainWindow):
    def __init__(
        self, fluorophore_images, cell_data, downscale_factor=4, annotated_data=None
    ):
        super().__init__()
        self.fluorophore_images = fluorophore_images
        self.cell_data = cell_data
        self.downscale_factor = downscale_factor
        self.downsampled_images = downscale_local_mean(
            fluorophore_images, (1, downscale_factor, downscale_factor)
        ).astype(fluorophore_images.dtype)
        self.current_fluorophore = 1
        self.zoom_level = None
        self.show_contours = True  # Flag for contour visibility

        # Ensure annotation columns exist
        for i in range(fluorophore_images.shape[0]):
            i = i + 1  # Channels are 1-indexed
            if f"barcode_channel_{i}" not in self.cell_data.properties:
                self.cell_data.properties[f"barcode_channel_{i}"] = 0

        if annotated_data is not None:
            self.load_existing_annotations(annotated_data)

        self.init_ui()

    def load_existing_annotations(self, annotated_data):
        """
        Loads existing annotations into cell data properties.
        This can handle either a path to a CSV file or a DataFrame directly.
        """
        if isinstance(annotated_data, str):
            self.cell_data.load_barcodes(barcode_path=annotated_data)
        elif isinstance(annotated_data, pd.DataFrame):
            self.cell_data.load_barcodes(barcodes=annotated_data)
        else:
            raise ValueError(
                "annotated_data should be a path to a CSV file or a DataFrame"
            )

    def init_ui(self):
        self.setWindowTitle("Cell Annotator")

        self.canvas = MplCanvas(self)
        self.img_display = self.canvas.ax.imshow(
            self.downsampled_images[self.current_fluorophore - 1],
            cmap="gray",
            vmin=0,
            vmax=65535,
        )
        self.canvas.ax.set_title(f"Fluorophore {self.current_fluorophore}")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_brightness)

        self.label = QLabel("Brightness")
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.save_button = QPushButton("Save Annotations")
        self.toggle_contours_checkbox = QCheckBox(
            "Show Contours"
        )  # Add checkbox for toggling contours

        self.prev_button.clicked.connect(self.prev_fluorophore)
        self.next_button.clicked.connect(self.next_fluorophore)
        self.save_button.clicked.connect(self.save_annotations)
        self.toggle_contours_checkbox.stateChanged.connect(
            self.toggle_contours
        )  # Connect checkbox to slot

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.prev_button)
        h_layout.addWidget(self.next_button)
        h_layout.addWidget(self.label)
        h_layout.addWidget(self.slider)
        h_layout.addWidget(self.save_button)
        h_layout.addWidget(self.toggle_contours_checkbox)  # Add checkbox to layout
        layout.addLayout(h_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.canvas.mpl_connect("scroll_event", self.onscroll)
        self.canvas.mpl_connect("button_press_event", self.onclick)
        self.draw_contours()

    def draw_contours(self):
        if self.zoom_level:
            cur_xlim = self.canvas.ax.get_xlim()
            cur_ylim = self.canvas.ax.get_ylim()
        else:
            cur_xlim = None
            cur_ylim = None

        # Clear the previous plot
        self.canvas.ax.clear()
        self.img_display = self.canvas.ax.imshow(
            self.downsampled_images[self.current_fluorophore - 1],  # Adjust indexing
            cmap="gray",
            vmin=0,
            vmax=self.slider.value() * 655.35,
        )
        self.canvas.ax.set_title(f"Fluorophore {self.current_fluorophore}")

        # Prepare to plot contours if visibility is enabled
        if self.show_contours:
            if not hasattr(self, "contours_cache"):
                self.contours_cache = {}

            for index, row in self.cell_data.properties.iterrows():
                mask_number = row["mask_number"]
                annotation = row[f"barcode_channel_{self.current_fluorophore}"]
                if mask_number != 0:
                    if mask_number not in self.contours_cache:
                        mask = self.cell_data.masks == mask_number
                        contours = find_contours(mask, 0.5)
                        self.contours_cache[mask_number] = {
                            "contours": contours,
                            "annotation": annotation,
                        }
                    else:
                        # Update annotation color in cache if it has changed
                        if self.contours_cache[mask_number]["annotation"] != annotation:
                            self.contours_cache[mask_number]["annotation"] = annotation

                    cached_contours = self.contours_cache[mask_number]["contours"]
                    color = (
                        "green"
                        if self.contours_cache[mask_number]["annotation"] == 1
                        else "red"
                    )
                    for contour in cached_contours:
                        contour = (
                            contour / self.downscale_factor
                        )  # Scale contours to downsampled image
                        self.canvas.ax.plot(
                            contour[:, 1], contour[:, 0], color=color, linewidth=1
                        )

        if cur_xlim and cur_ylim:
            self.canvas.ax.set_xlim(cur_xlim)
            self.canvas.ax.set_ylim(cur_ylim)

        self.canvas.draw()

    def toggle_contours(self, state):
        self.show_contours = (
            state == Qt.Checked
        )  # Update visibility based on checkbox state
        self.draw_contours()  # Redraw contours based on new visibility setting

    def onclick(self, event):
        x, y = int(event.xdata * self.downscale_factor), int(
            event.ydata * self.downscale_factor
        )  # Scale coordinates to original image
        cell_id = self.get_cell_id(x, y)
        if cell_id:
            self.toggle_annotation(cell_id)
            self.draw_contours()

    def onscroll(self, event):
        self.zoom_level = self.canvas.ax.get_xlim(), self.canvas.ax.get_ylim()

        cur_xlim = self.canvas.ax.get_xlim()
        cur_ylim = self.canvas.ax.get_ylim()

        xdata = event.xdata
        ydata = event.ydata

        scale_factor = 1 / 1.2 if event.button == "up" else 1.2

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.canvas.ax.set_xlim(
            [xdata - new_width * (1 - relx), xdata + new_width * relx]
        )
        self.canvas.ax.set_ylim(
            [ydata - new_height * (1 - rely), ydata + new_height * rely]
        )
        self.canvas.draw()

    def toggle_annotation(self, cell_id):
        current_value = self.cell_data.properties.loc[
            self.cell_data.properties["mask_number"] == cell_id,
            f"barcode_channel_{self.current_fluorophore}",
        ]
        new_value = 1 if current_value.empty or current_value.values[0] == 0 else 0
        self.cell_data.properties.loc[
            self.cell_data.properties["mask_number"] == cell_id,
            f"barcode_channel_{self.current_fluorophore}",
        ] = new_value
        print(
            f"Cell {cell_id} for fluorophore {self.current_fluorophore} annotated as {'positive' if new_value == 1 else 'negative'}"
        )

    def next_fluorophore(self):
        self.current_fluorophore = (
            self.current_fluorophore % self.downsampled_images.shape[0]
        ) + 1
        self.update_display()

    def prev_fluorophore(self):
        self.current_fluorophore = (
            self.current_fluorophore - 2
        ) % self.downsampled_images.shape[0] + 1
        self.update_display()

    def update_display(self):
        if self.zoom_level:
            cur_xlim, cur_ylim = self.canvas.ax.get_xlim(), self.canvas.ax.get_ylim()
        else:
            cur_xlim, cur_ylim = None, None

        self.img_display.set_data(
            self.downsampled_images[self.current_fluorophore - 1]
        )  # Adjust indexing
        self.canvas.ax.set_title(f"Fluorophore {self.current_fluorophore}")
        if cur_xlim and cur_ylim:
            self.canvas.ax.set_xlim(cur_xlim)
            self.canvas.ax.set_ylim(cur_ylim)
        self.draw_contours()

    def update_brightness(self, value):
        self.img_display.set_clim(0, value * 655.35)
        self.canvas.draw()

    def get_cell_id(self, x, y):
        mask_number = self.cell_data.masks[y, x]
        return mask_number if mask_number != 0 else None

    def save_annotations(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotations",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options,
        )
        if file_name:
            self.cell_data.properties.to_csv(file_name, index=False)

class InteractiveThresholding(QMainWindow):
    def __init__(
        self,
        original_image,
        thresholding_images,
        binarizer,
        use_exponential_slider=False,
    ):
        super().__init__()
        self.original_image = original_image
        self.thresholding_images = thresholding_images
        self.binarizer = binarizer
        self.use_exponential_slider = use_exponential_slider

        self.initial_thresholds = [
            threshold_otsu(self.thresholding_images[ch].flatten())
            for ch in range(self.thresholding_images.shape[0])
        ]
        self.thresholds = self.initial_thresholds.copy()
        self.current_channel = 0

        self.show_positive = True
        self.show_all = True
        self.fluorophore_absent = [False] * self.thresholding_images.shape[0]

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        self.precompute_contours()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Interactive Thresholding")
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        self.min_val = np.min(self.thresholding_images)
        self.max_val = np.max(self.thresholding_images)

        self.slider = QSlider(Qt.Horizontal)
        if self.use_exponential_slider:
            self.log_min_val = np.log10(self.min_val + 1e-5)
            self.log_max_val = np.log10(self.max_val + 0.01 * self.max_val)
            self.slider.setRange(
                int(self.log_min_val * 1000), int(self.log_max_val * 1000)
            )
            self.slider.setValue(
                int(np.log10(self.thresholds[self.current_channel] + 1e-5) * 1000)
            )
        else:
            self.slider.setRange(int(self.min_val * 1000), int(self.max_val * 1000))
            self.slider.setValue(int(self.thresholds[self.current_channel] * 1000))

        self.slider.valueChanged.connect(self.update_threshold)
        main_layout.addWidget(self.slider)

        toggle_layout = QHBoxLayout()
        self.toggle_positive = QCheckBox("Show Positive")
        self.toggle_positive.setChecked(True)
        self.toggle_positive.stateChanged.connect(self.toggle_positive_outlines)
        toggle_layout.addWidget(self.toggle_positive)

        self.toggle_all = QCheckBox("Show All")
        self.toggle_all.setChecked(True)
        self.toggle_all.stateChanged.connect(self.toggle_all_outlines)
        toggle_layout.addWidget(self.toggle_all)

        self.fluorophore_absent_checkbox = QCheckBox("Fluorophore not present")
        self.fluorophore_absent_checkbox.stateChanged.connect(self.toggle_fluorophore_absence)
        toggle_layout.addWidget(self.fluorophore_absent_checkbox)

        main_layout.addLayout(toggle_layout)

        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous Channel")
        self.btn_prev.clicked.connect(self.prev_channel)
        btn_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Channel")
        self.btn_next.clicked.connect(self.next_channel)
        btn_layout.addWidget(self.btn_next)

        main_layout.addLayout(btn_layout)

        self.plot_channel(self.current_channel)

    def precompute_contours(self):
        self.positive_contours = [None] * self.thresholding_images.shape[0]
        self.all_contours = []
        for mask_number in np.unique(self.binarizer.cell_data.masks):
            if mask_number == 0:
                continue
            mask = self.binarizer.cell_data.masks == mask_number
            contours = find_contours(mask, 0.5)
            self.all_contours.extend(contours)

        for ch in range(self.thresholding_images.shape[0]):
            self.update_positive_contours(ch)

    def update_positive_contours(self, channel):
        threshold = self.thresholds[channel]
        binary_mask = self.binarizer.binarize_channels(channel=channel, thresholds=[threshold])[0, ...]
        positive_mask = binary_mask > 0
        self.positive_contours[channel] = find_contours(positive_mask, 0.5)

    def plot_channel(self, channel):
        self.ax.clear()
        original_signal = self.original_image[channel, ...]
        vmax = np.percentile(original_signal, 99.9)
        self.ax.imshow(original_signal, cmap="gray", vmax=vmax)
        self.ax.set_title(f"Channel {channel + 1}")
        self.ax.axis("off")

        if self.show_all and self.all_contours:
            lines_all = [c[:, [1, 0]] for c in self.all_contours]
            self.ax.add_collection(LineCollection(lines_all, colors="lightblue", linewidths=1))

        if self.show_positive and self.positive_contours[channel]:
            lines_positive = [c[:, [1, 0]] for c in self.positive_contours[channel]]
            self.ax.add_collection(LineCollection(lines_positive, colors="green", linewidths=1))

        self.canvas.draw()

    def update_threshold(self):
        if self.fluorophore_absent[self.current_channel]:
            self.thresholds[self.current_channel] = np.inf
            self.positive_contours[self.current_channel] = []
        else:
            val = self.slider.value() / 1000.0
            self.thresholds[self.current_channel] = 10**val - 1e-5 if self.use_exponential_slider else val
            self.update_positive_contours(self.current_channel)
        self.plot_channel(self.current_channel)

    def toggle_positive_outlines(self, state):
        self.show_positive = state == Qt.Checked
        self.plot_channel(self.current_channel)

    def toggle_all_outlines(self, state):
        self.show_all = state == Qt.Checked
        self.plot_channel(self.current_channel)

    def toggle_fluorophore_absence(self, state):
        self.fluorophore_absent[self.current_channel] = state == Qt.Checked
        self.update_threshold()

    def next_channel(self):
        self.current_channel = (self.current_channel + 1) % self.thresholding_images.shape[0]
        self.update_channel()

    def prev_channel(self):
        self.current_channel = (self.current_channel - 1) % self.thresholding_images.shape[0]
        self.update_channel()

    def update_channel(self):
        self.fluorophore_absent_checkbox.blockSignals(True)
        self.fluorophore_absent_checkbox.setChecked(self.fluorophore_absent[self.current_channel])
        self.fluorophore_absent_checkbox.blockSignals(False)

        slider_val = self.max_val if self.fluorophore_absent[self.current_channel] else self.thresholds[self.current_channel]
        self.slider.blockSignals(True)
        self.slider.setValue(int(slider_val * 1000))
        self.slider.blockSignals(False)

        self.update_positive_contours(self.current_channel)
        self.plot_channel(self.current_channel)


if __name__ == "__main__":
    import sys

    # Example usage:
    app = QApplication(sys.argv)
    original_image = np.random.rand(3, 100, 100)  # Example image
    thresholding_images = np.random.rand(3, 100, 100)  # Example thresholding images
    binarizer = None  # Replace with actual binarizer instance

    # For demo purposes
    class DummyBinarizer:
        def binarize_channels(self, channel, thresholds):
            return np.random.rand(1, 100, 100) > thresholds[0]

    binarizer = DummyBinarizer()

    window = InteractiveThresholding(original_image, thresholding_images, binarizer)
    window.show()
    sys.exit(app.exec_())
