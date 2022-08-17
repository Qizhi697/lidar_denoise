#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
from vispy import io, scene
import numpy as np
from matplotlib import pyplot as plt
from utils.laserscan import LaserScan, SemLaserScan
import win32gui, win32ui, win32con, win32api

def window_capture(filename):
    hwnd = 0  # 窗口的编号，0号表示当前活跃窗口
    # 根据窗口句柄获取窗口的设备上下文DC（Divice Context）
    hwndDC = win32gui.GetWindowDC(hwnd)
    # 根据窗口的DC获取mfcDC
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    # mfcDC创建可兼容的DC
    saveDC = mfcDC.CreateCompatibleDC()
    # 创建bigmap准备保存图片
    saveBitMap = win32ui.CreateBitmap()
    # 获取监控器信息
    MoniterDev = win32api.EnumDisplayMonitors(None, None)
    w = MoniterDev[0][2][2]
    h = MoniterDev[0][2][3]
    # print w,h　　　#图片大小
    # 为bitmap开辟空间
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    # 高度saveDC，将截图保存到saveBitmap中
    saveDC.SelectObject(saveBitMap)
    # 截取从左上角（0，0）长宽为（w，h）的图片
    saveDC.BitBlt((0, -35), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
    saveBitMap.SaveBitmapFile(saveDC, filename)


class LaserScanVis:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, scan, scan_names, label_names, prelabel_names, semantics=True, instances=False, show_clear=False,
                 show_denoised=False):
        self.scan = scan
        self.scan_names = scan_names
        self.label_names = label_names
        self.prelabel_names = prelabel_names
        self.offset = 394
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
        self.canvas = SceneCanvas(keys='interactive', show=True, bgcolor='white')
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()
        # cam1 = scene.cameras.TurntableCamera(elevation=30, azimuth=-30, roll=0, distance=18)
        # cam2 = scene.cameras.TurntableCamera(elevation=45, azimuth=-45, roll=0, distance=60)
        cam2 = scene.cameras.TurntableCamera(elevation=45, azimuth=-45, roll=0, distance=100)

        # laserscan part
        # self.scan_view = vispy.scene.widgets.ViewBox(
        #     border_color='white', parent=self.canvas.scene, camera=cam2)
        # self.grid.add_widget(self.scan_view, 0, 0)
        # self.scan_vis = visuals.Markers()
        # self.scan_view.camera = 'turntable'
        # self.scan_view.add(self.scan_vis)
        # visuals.XYZAxis(parent=self.scan_view.scene)


        if self.semantics:
            print("Using semantics in visualizer")
            self.sem_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene, camera=cam2)
            self.grid.add_widget(self.sem_view, 0, 0)
            self.sem_vis = visuals.Markers()
            self.sem_view.camera = 'turntable'
            self.sem_view.add(self.sem_vis)
            # visuals.XYZAxis(parent=self.sem_view.scene)

        # self.sem_view.camera.link(self.scan_view.camera)

        if self.show_clear:
            print("Show denoised in visualizer")
            self.clear_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.canvas.scene, camera=cam2)
            self.grid.add_widget(self.clear_view, 0, 1)
            self.clear_vis = visuals.Markers()
            self.clear_view.camera = 'turntable'
            self.clear_view.add(self.clear_vis)

        if self.show_denoised:
            print("Show denoised in visualizer")
            self.denoised_view = vispy.scene.widgets.ViewBox(
                border_color='white', parent=self.canvas.scene, camera=cam2)
            self.grid.add_widget(self.denoised_view, 0, 2)
            self.denoised_vis = visuals.Markers()
            self.denoised_view.camera = 'turntable'
            self.denoised_view.add(self.denoised_vis)

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
            # self.scan.open_label(self.prelabel_names[self.offset])
        if self.show_denoised:
            self.scan.open_prelabel(self.prelabel_names[self.offset])
        self.scan.colorize()

        # then change names
        notation = "[raw,clear,denoised]"
        title = "scan " + str(self.offset) + notation
        self.canvas.title = title
        self.img_canvas.title = title

        # then do all the point cloud stuff
        # power = 16
        # # print()
        # range_data = np.copy(self.scan.unproj_range)
        # # print(range_data.max(), range_data.min())
        # range_data = range_data ** (1 / power)
        # # print(range_data.max(), range_data.min())
        # viridis_range = ((range_data - range_data.min()) /
        #                  (range_data.max() - range_data.min()) *
        #                  255).astype(np.uint8)
        # viridis_map = self.get_mpl_colormap("viridis")
        # viridis_colors = viridis_map[viridis_range]
        # self.scan_vis.set_data(self.scan.points,
        #                        face_color=viridis_colors[..., ::-1],
        #                        edge_color=viridis_colors[..., ::-1],
        #                        size=1)

        # plot semantics
        if self.semantics:
            self.sem_vis.set_data(self.scan.points,
                                  face_color=self.scan.sem_label_color[..., ::-1],
                                  edge_color=self.scan.sem_label_color[..., ::-1],
                                  size=1)

        if self.show_clear:
            self.clear_vis.set_data(self.scan.points,
                                    face_color=self.scan.sem_label_clearcolor[..., ::-1],
                                    edge_color=self.scan.sem_label_clearcolor[..., ::-1],
                                    size=0.2)

        if self.show_denoised:
            vis_point_index = (self.scan.sem_prelabel == 0) | (self.scan.sem_prelabel == 1)
            vis_point_index=np.concatenate((vis_point_index,np.ones(len(self.scan.points)-len(vis_point_index)).astype(np.bool)),0)
            self.denoised_vis.set_data(self.scan.points[vis_point_index],
                                       face_color=self.scan.sem_label_denoisedcolor[vis_point_index, ::-1],
                                       edge_color=self.scan.sem_label_denoisedcolor[vis_point_index, ::-1],
                                       size=1)

        # img = self.canvas.render()
        png_name = "png/our/scan " + str(self.offset) + ".png"
        window_capture(png_name)
        # io.write_png(png_name, img)

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
