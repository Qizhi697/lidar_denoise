#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
from utils.laserscan import LaserScan, SemLaserScan


class LaserScanVis:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, scan, scan_names, label_names, prelabel_names, semantics=True, instances=False, show_clear=False, show_denoised=False):
        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.prelabel_names = prelabel_names
        self.offset = 0
        self.total = len(self.scan_names)
        self.semantics = semantics
        self.show_clear = show_clear
        self.show_denoised = show_denoised
        self.reset()
        self.update_scan()

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        if self.semantics:
            print("Using semantics in visualizer")
            self.sem_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.canvas.scene)
            self.grid.add_widget(self.sem_view, 0, 0)
            self.sem_vis = visuals.Markers()
            self.sem_view.camera = 'turntable'
            self.sem_view.add(self.sem_vis)
            visuals.XYZAxis(parent=self.sem_view.scene)

        # self.sem_view.camera.link(self.scan_view.camera)

        if self.show_clear:
            print("Show denoised in visualizer")
            self.clear_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.canvas.scene)
            self.grid.add_widget(self.clear_view, 0, 1)
            self.clear_vis = visuals.Markers()
            self.clear_view.camera = 'turntable'
            self.clear_view.add(self.clear_vis)
            visuals.XYZAxis(parent=self.clear_view.scene)

        if self.show_denoised:
            print("Show denoised in visualizer")
            self.denoised_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.canvas.scene)
            self.grid.add_widget(self.denoised_view, 0, 2)
            self.denoised_vis = visuals.Markers()
            self.denoised_view.camera = 'turntable'
            self.denoised_view.add(self.denoised_vis)
            visuals.XYZAxis(parent=self.denoised_view.scene)

        # img canvas size
        self.multiplier = 1
        self.canvas_W = 1024
        self.canvas_H = 64
        if self.semantics:
            self.multiplier += 1
        if self.show_clear:
            self.multiplier += 1
        if self.show_denoised:
            self.multiplier += 1

        # new canvas for img
        self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                      size=(self.canvas_W, self.canvas_H * self.multiplier))
        # grid
        self.img_grid = self.img_canvas.central_widget.add_grid()
        # interface (n next, b back, q quit, very simple)
        self.img_canvas.events.key_press.connect(self.key_press)
        self.img_canvas.events.draw.connect(self.draw)

        # add semantics
        if self.semantics:
            self.sem_img_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.img_canvas.scene)
            self.img_grid.add_widget(self.sem_img_view, 0, 0)
            self.sem_img_vis = visuals.Image(cmap='viridis')
            self.sem_img_view.add(self.sem_img_vis)

        if self.show_clear:
            self.clear_img_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.img_canvas.scene)
            self.img_grid.add_widget(self.clear_img_view, 1, 0)
            self.clear_img_vis = visuals.Image(cmap='viridis')
            self.clear_img_view.add(self.clear_img_vis)

        if self.show_denoised:
            self.denoised_img_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.img_canvas.scene)
            self.img_grid.add_widget(self.denoised_img_view, 2, 0)
            self.denoised_img_vis = visuals.Image(cmap='viridis')
            self.denoised_img_view.add(self.denoised_img_vis)


    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def update_scan(self):
        # first open data
        self.scan.open_scan(self.scan_names[self.offset])
        if self.semantics:
            self.scan.open_label(self.label_names[self.offset])
        if self.show_denoised:
            self.scan.open_prelabel(self.prelabel_names[self.offset])
        self.scan.colorize()

        # then change names
        notation = "[raw,clear,denoised]"
        title = "scan " + str(self.offset) + notation
        self.canvas.title = title
        self.img_canvas.title = title

        # then do all the point cloud stuff

        # plot semantics
        if self.semantics:
            self.sem_vis.set_data(self.scan.points,
                                  face_color=self.scan.sem_label_color[..., ::-1],
                                  edge_color=self.scan.sem_label_color[..., ::-1],
                                  size=0.1)

        if self.show_clear:
            self.clear_vis.set_data(self.scan.points,
                                   face_color=self.scan.sem_label_clearcolor[..., ::-1],
                                   edge_color=self.scan.sem_label_clearcolor[..., ::-1],
                                   size=0.1)

        if self.show_denoised:
            self.denoised_vis.set_data(self.scan.points,
                                   face_color=self.scan.sem_label_denoisedcolor[..., ::-1],
                                   edge_color=self.scan.sem_label_denoisedcolor[..., ::-1],
                                   size=0.1)


        # now do all the range image stuff

        if self.semantics:
            self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
            self.sem_img_vis.update()

        if self.show_clear:
            self.clear_img_vis.set_data(self.scan.proj_clear_color[..., ::-1])
            self.clear_img_vis.update()

        if self.show_denoised:
            self.denoised_img_vis.set_data(self.scan.proj_denoised_color[..., ::-1])
            self.denoised_img_vis.update()

    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()
        self.img_canvas.events.key_press.block()
        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0
            self.update_scan()
        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1
            self.update_scan()
        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
            self.img_canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.img_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()
