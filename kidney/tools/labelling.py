import glob
import os
from typing import Tuple, Optional, List

import ipywidgets as widgets
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from skimage.segmentation import flood

from kidney.utils.tiff import read_tiff


def zoom_factory(ax: Axes, base_scale: float = 1.1):

    def limits_range(lim: Tuple) -> float:
        return lim[1] - lim[0]

    fig = ax.get_figure()
    toolbar = fig.canvas.toolbar
    toolbar.push_current()

    old_x_lim = ax.get_xlim()
    old_y_lim = ax.get_ylim()
    old_x_range = limits_range(old_x_lim)
    old_y_range = limits_range(old_y_lim)
    old_center = (old_x_lim[0] + old_x_lim[1])/2, (old_y_lim[0] + old_y_lim[1])/2

    def zoom_callback(event):
        nonlocal ax
        curr_x_lim = ax.get_xlim()
        curr_y_lim = ax.get_ylim()
        x_data, y_data = event.xdata, event.ydata
        if event.button == "up":
            scale_factor = base_scale
        elif event.button == "down":
            scale_factor = 1/base_scale
        else:
            scale_factor = 1
        new_x_lim = [
            x_data - (x_data - curr_x_lim[0])/scale_factor,
            x_data + (curr_x_lim[1] - x_data)/scale_factor
        ]
        new_y_lim = [
            y_data - (y_data - curr_y_lim[0])/scale_factor,
            y_data + (curr_y_lim[1] - y_data)/scale_factor
        ]
        new_y_range = limits_range(new_y_lim)
        new_x_range = limits_range(new_x_lim)
        if np.abs(new_y_range) > np.abs(old_y_range):
            new_y_lim = (
                old_center[1] - new_y_range/2,
                old_center[1] + new_y_range/2
            )
        if np.abs(new_x_range) > np.abs(old_x_range):
            new_x_lim = (
                old_center[0] - new_x_range/2,
                old_center[0] + new_x_range/2
            )
        ax.set_xlim(new_x_lim)
        ax.set_ylim(new_y_lim)
        toolbar.push_current()
        ax.figure.canvas.draw_idle()

    call_id = fig.canvas.mpl_connect("scroll_event", zoom_callback)

    def disconnect_zoom():
        fig.canvas.mpl_disconnect(call_id)

    return disconnect_zoom


class PanHandler:

    def __init__(self, figure: Figure):
        self.figure = figure
        self._id_drag = None
        self._xy_press = []
        self._button_pressed = 0

    def press(self, event):
        if event.button == 1:
            return
        elif event.button == 3:
            self._button_pressed = 1
        else:
            self._cancel_action()
            return
        x, y = event.x, event.y
        self._xy_press = []
        for i, a in enumerate(self.figure.get_axes()):
            if (
                x is not None and
                y is not None and
                a.in_axes(event) and
                a.get_navigate() and
                a.can_pan()
            ):
                a.start_pan(x, y, event.button)
                self._xy_press.append((a, i))
                self._id_drag = self.figure.canvas.mpl_connect(
                    "motion_notify_event",
                    self._mouse_move
                )

    def release(self, event):
        self._cancel_action()
        for a, _ in self._xy_press:
            a.end_pan()
        self._cancel_action()

    def _cancel_action(self):
        self._xy_press = []
        if self._id_drag:
            self.figure.canvas.mpl_disconnect(self._id_drag)
            self._id_drag = None

    def _mouse_move(self, event):
        for a, _ in self._xy_press:
            a.drag_pan(1, event.key, event.x, event.y)
        self.figure.canvas.draw_idle()


class ImageSegmentation:
    # https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
    # https://github.com/ianhi/AC295-final-project-JWI/blob/master/lib/labelling.py

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        classes: Optional[List] = None,
        colors: Optional[List] = None,
        overlay_alpha: float = 0.5,
        figsize: Tuple[int, int] = (10, 10),
        scroll_to_zoom: bool = True,
        zoom_scale: float = 1.1,
    ):
        assert os.path.exists(images_dir)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        os.makedirs(self.masks_dir, exist_ok=True)

        plt.ioff()
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.gca()
        self.lasso = LassoSelector(
            ax=self.ax,
            onselect=self.on_select,
            lineprops={"color": "black", "linewidth": 1, "alpha": 0.8},
            button=1,
            useblit=False
        )
        self.lasso.set_visible(True)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("button_release_event", self._release)
        self.pan_handler = PanHandler(self.fig)
        self.image_paths = glob.glob(os.path.join(self.images_dir, "*.png"))
        plt.ion()

        n_classes = len(classes) if classes is not None else 1
        colors = colors if colors is not None else ["red", "violet", "blue"]
        color_map = matplotlib.colors.LinearSegmentedColormap.from_list("segments", colors)
        self.colors = np.vstack([[0, 0, 0], color_map(np.arange(n_classes))[:, :3]])

        if n_classes > 1:
            pass  # dropdown implementation goes there

        self.curr_image = None
        self.indexes = None
        self.total_images = len(self.image_paths)
        self.image = None
        self.image_shape = None
        self.mask_path = None
        self.pix = None
        self.displayed = None
        self.class_mask = None
        self.vertices = None

        self.lasso_button = widgets.Button(
            description="Lasso Select",
            button_style="success",
            icon="mouse-pointer"
        )
        self.flood_button = widgets.Button(
            description="Flood Fill",
            button_style="",
            icon="fill-drip"
        )
        self.erase_check_box = widgets.Checkbox(
            value=False,
            description="Erase Mode",
            indent=False
        )
        self.reset_button = widgets.Button(
            description="Reset",
            button_style="",
            icon="refresh"
        )
        self.save_button = widgets.Button(
            description="Save Mask",
            button_style="",
            icon="floppy-o"
        )
        self.next_button = widgets.Button(
            description="Next Image",
            button_style="",
            icon="arrow-right"
        )
        self.prev_button = widgets.Button(
            description="Previous Image",
            button_style="",
            icon="arrow-left",
            disabled=True
        )

        self.reset_button.on_click(self.reset)
        self.save_button.on_click(self.save_mask)
        self.next_button.on_click(self._change_image)
        self.prev_button.on_click(self._change_image)

        def button_click(button: widgets.Button):
            nonlocal self
            if button is self.flood_button == "Flood Fill":
                self.flood_button.button_style = "success"
                self.lasso_button.button_style = ""
                self.lasso.set_active(False)
            else:
                self.flood_button.button_style = ""
                self.lasso_button.button_style = "success"
                self.lasso.set_active(True)

        self.lasso_button.on_click(button_click)
        self.flood_button.on_click(button_click)
        self.overlay_alpha = overlay_alpha
        self.new_image(0)

        if zoom_scale is not None:
            self.disconnected_scroll = zoom_factory(self.ax, zoom_scale)

    def new_image(self, index: int):
        image_path = self.image_paths[index]
        filename = os.path.basename(image_path)
        self.image = self.read_image(image_path)
        self.curr_image = index
        self.ax.set_title(filename)
        self.mask_path = os.path.join(self.masks_dir, f"seg.{filename}")

        if self.image_shape is None or self.image.shape != self.image_shape:
            self.image_shape = self.image.shape
            pix_x = np.arange(self.image_shape[0])
            pix_y = np.arange(self.image_shape[1])
            xs, ys = np.meshgrid(pix_y, pix_x)
            self.pix = np.vstack((xs.flatten(), ys.flatten())).T
            self.displayed = self.ax.imshow(self.image)
            self.fig.canvas.toolbar._nav_stack.clear()
            self.fig.canvas.toolbar.push_current()
            if os.path.exists(self.mask_path):
                self.class_mask = read_tiff(self.mask_path)
            else:
                self.class_mask = np.zeros([
                    self.image_shape[0], self.image_shape[1]
                ], dtype=np.uint8)

        else:
            self.displayed.set_data(self.image)
            if os.path.exists(self.mask_path):
                self.class_mask = read_tiff(self.mask_path)
            else:
                self.class_mask[:, :] = 0
            self.fig.canvas.toolbar.home()

        self.update_array()

    def read_image(self, path: str) -> np.ndarray:
        return np.random.randint(0, 2, size=(1024, 1024))

    def reset(self, *args):
        if self.displayed is not None and self.image is not None:
            self.displayed.set_data(self.image)
            self.class_mask[:, :] = -1
            self.fig.canvas.draw()

    def on_click(self, event):
        if event.button == 1:
            if event.xdata is not None and not self.lasso.active:
                self.indexes = flood(
                    self.class_mask,
                    (np.int(event.ydata), np.int(event.xdata))
                )
                self.update_array()
        elif event.button == 3:
            self.pan_handler.press(event)

    def update_array(self):
        if self.displayed is None:
            return
        arr = self.displayed.get_array().data
        if self.erase_check_box.value:
            if self.indexes is not None:
                self.class_mask[self.indexes] = 0
                arr[self.indexes] = self.image[self.indexes]
        elif self.indexes is not None:
            self.class_mask[self.indexes] = 1
            overlay = self.colors[self.class_mask[self.indexes]]*255*self.overlay_alpha
            arr[self.indexes] = overlay + self.image[self.indexes]*(1 - self.overlay_alpha)
        else:
            index = self.class_mask != 0
            if index.any():
                overlay = self.colors[self.class_mask[index]]*255*self.overlay_alpha
                arr[index] = overlay + self.image[index]*(1 - self.overlay_alpha)
        self.displayed.set_data(arr)

    def on_select(self, vertices):
        self.vertices = vertices
        self.indexes = (
            Path(self.vertices)
            .contains_points(self.pix, radius=0)
            .reshape(450, 450)
        )
        self.update_array()
        self.fig.canvas.draw_idle()

    def render(self):
        return widgets.VBox([
            widgets.HBox([self.lasso_button, self.flood_button]),
            widgets.HBox([self.reset_button, self.erase_check_box]),
            self.fig.canvas,
            widgets.HBox([self.save_button, self.prev_button, self.next_button])
        ])

    def save_mask(self, save_if_non_zero=False):
        if save_if_non_zero or np.any(self.class_mask != 0):
            print("saved!")

    def _change_image(self, button: widgets.Button):
        if button is self.next_button:
            self.curr_image = min(self.curr_image+1, self.total_images-1)
            button.disabled = self.curr_image == self.total_images - 1

        elif button is self.prev_button:
            self.curr_image = max(self.curr_image-1, 0)
            button.disabled = self.curr_image == 0

        self.save_mask()
        self.new_image(self.curr_image)

    def _release(self, event):
        self.pan_handler.release(event)

    def _ipython_display_(self):
        display(self.render())
