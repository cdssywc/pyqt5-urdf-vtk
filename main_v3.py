# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# Copyright (c) 2025, Chen XingYu. All rights reserved.

# License: Non-Commercial Use Only / 仅限非商业使用
# -----------------------------------------------------------------------------
# 本代码及其衍生作品仅允许用于个人学习、学术研究与教学等非商业场景。
# 严禁任何形式的商业使用，包括但不限于：出售、付费服务、SaaS/在线服务、
# 广告变现、集成到商业产品或用于商业咨询/竞赛/投标等。如需商业授权，请
# 先行获得版权所有者书面许可并签署授权协议。

# 允许的非商业使用条件：
# 1) 保留本版权与许可声明；2) 在衍生作品/发表物中进行署名（Lu Yaoheng）并
# 标明来源仓库；3) 不得移除或修改本段声明。

# 免责声明：本代码按“现状”提供，不含任何明示或默示担保。作者不对因使用本
# 代码产生的任何直接或间接损失承担责任。使用者需自行评估并承担风险。

# English Summary:
# This code is provided for personal, academic, and research purposes only.
# Any commercial use (sale, paid service, SaaS, ad-monetization, integration
# into commercial products, consultancy, competitions, bids, etc.) is strictly
# prohibited without prior written permission from the copyright holder.
# Keep this notice intact and provide proper attribution in derived works.
# Provided "as is" without warranty of any kind. Use at your own risk.

# Contact / 商务与授权联系: <cdssywc@163.com>
"""
URDF 机械臂查看器（Qt5 + VTK，增强控件 + 视图预设 + 路径绘制 + 球体添加 + 坐标系显示 + 球体双击编辑）
- 一次加载 STL -> VTK Actor；滑条仅更新变换矩阵（高性能）
- 显示设置：背景浅蓝切换、地面网格开关、主/辅光亮度调节、视图预设（正/侧/顶/斜）
- 路径绘制：可勾选显示末端运动路径，可清除路径
- 球体添加：可设置球体大小和位置（世界坐标）
- 坐标系显示：基座/末端坐标系（URDF），可调整长度
- 双击球体：选中球体，显示其坐标系；可改半径与位置；选择轴并沿轴移动
"""

import os, math, numpy as np, xml.etree.ElementTree as ET
from anytree import Node, RenderTree
import transformations as tf
import trimesh

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QSlider, QGroupBox, QScrollArea, QSizePolicy, QPushButton, QCheckBox, QToolButton,
    QHBoxLayout as QHBox, QLineEdit, QDoubleSpinBox, QSpinBox, QFormLayout, QComboBox,
    QGridLayout, QTabWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QColor

import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# =============== 路径配置 ===============
BASE_DIR = r"C:\Users\15973\Desktop\滚落的狮子-6轴机械臂\urdf4"   # 按你当前的目录
URDF_CANDIDATES = [
    os.path.join(BASE_DIR, "urdf4.urdf"),
    os.path.join(BASE_DIR, "urdf", "urdf4.urdf"),
]
GLOBAL_SCALE = 1.0   # STL 为毫米->米请设 0.001

# 全局关节位姿（弧度或米）
JOINT_POS = {}

# =============== 工具函数 ===============
def ensure_urdf_exists():
    for p in URDF_CANDIDATES:
        if os.path.isfile(p): return p
    raise FileNotFoundError("未找到 URDF：\n  - " + "\n  - ".join(URDF_CANDIDATES))

def parse_xyz(text, default=(0.0,0.0,0.0)):
    if not text: return np.array(default, float)
    parts = [float(x) for x in text.replace(",", " ").split()]
    return np.array(parts[:3] if len(parts)>=3 else default, float)

def parse_rpy(text, default=(0.0,0.0,0.0)): return parse_xyz(text, default)
def rpy_matrix(rpy): return tf.euler_matrix(*rpy, axes='sxyz')
def T_from_xyz_rpy(xyz, rpy): return tf.concatenate_matrices(tf.translation_matrix(xyz), rpy_matrix(rpy))

def rotation_about_axis(theta, axis):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis) or 1.0
    return tf.rotation_matrix(theta, axis/n)

def translation_along_axis(dist, axis):
    axis = np.asarray(axis, float)
    return tf.translation_matrix(axis*float(dist))

def resolve_mesh_path(mesh_filename):
    if not mesh_filename: return None
    if mesh_filename.startswith("package://"):
        parts = mesh_filename.split('/', 3)
        mesh_rel = parts[3] if len(parts)>=4 else mesh_filename.replace("package://","")
    else:
        mesh_rel = mesh_filename
    cands = [
        os.path.join(BASE_DIR, mesh_rel),
        os.path.join(BASE_DIR, "meshes", mesh_rel),
        os.path.join(BASE_DIR, "urdf", mesh_rel),
        os.path.join(BASE_DIR, os.path.basename(mesh_rel)),
    ]
    if mesh_rel.lower().startswith("meshes/"):
        cands.append(os.path.join(BASE_DIR, mesh_rel.split("/",1)[1]))
    for c in cands:
        if os.path.isfile(c): return c
    return None

# =============== URDF 解析 ===============
class URDFRobot:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        self.links = {}    # name -> {"visuals":[{filename,xyz,rpy,scale}]}
        self.joints = {}   # name -> {"type","parent","child","xyz","rpy","axis","limit":(lower,upper)}
        self.root_link = None
        self.adj = {}
        self.root_node = None
        self.link_T = {}
        self.joint_order = []
        self.end_effector_link = None  # 末端执行器链接
        self.parse_urdf()
        self.build_tree()
        self.find_end_effector()

    def parse_urdf(self):
        root = ET.parse(self.urdf_path).getroot()

        # links
        for link in root.findall("link"):
            lname = link.get("name")
            visuals=[]
            for vis in link.findall("visual"):
                vo = vis.find("origin")
                v_xyz = parse_xyz(vo.get("xyz") if vo is not None else None)
                v_rpy = parse_rpy(vo.get("rpy") if vo is not None else None)
                mesh = vis.find("geometry/mesh")
                if mesh is not None:
                    fname = mesh.get("filename")
                    scale = parse_xyz(mesh.get("scale") if mesh.get("scale") else "1 1 1")
                    visuals.append({"filename":fname,"xyz":v_xyz,"rpy":v_rpy,"scale":scale})
            self.links[lname] = {"visuals":visuals}

        # joints
        for joint in root.findall("joint"):
            jn = joint.get("name")
            jt = joint.get("type","fixed").lower()
            parent = joint.find("parent").get("link")
            child  = joint.find("child").get("link")
            jo = joint.find("origin")
            j_xyz = parse_xyz(jo.get("xyz") if jo is not None else None)
            j_rpy = parse_rpy(jo.get("rpy") if jo is not None else None)
            ax = joint.find("axis")
            axis = parse_xyz(ax.get("xyz") if ax is not None else "1 0 0")
            # limits
            lower, upper = -math.pi, math.pi
            lim = joint.find("limit")
            if lim is not None:
                if lim.get("lower") is not None: lower = float(lim.get("lower"))
                if lim.get("upper") is not None: upper = float(lim.get("upper"))
            self.joints[jn] = {"type":jt,"parent":parent,"child":child,"xyz":j_xyz,"rpy":j_rpy,"axis":axis,"limit":(lower,upper)}
            self.adj.setdefault(parent,[]).append((jn,child))
            self.joint_order.append(jn)

        # root link
        all_links = set(self.links.keys())
        children = {j["child"] for j in self.joints.values()}
        roots = list(all_links - children)
        self.root_link = roots[0] if roots else ("base_link" if "base_link" in all_links else next(iter(all_links)))

    def build_tree(self):
        nodes = {self.root_link: Node(self.root_link)}
        stack=[self.root_link]
        while stack:
            pl=stack.pop()
            for (jn,cl) in self.adj.get(pl,[]):
                if cl not in nodes:
                    nodes[cl]=Node(cl,parent=nodes[pl]); stack.append(cl)
        self.root_node = nodes[self.root_link]

    def find_end_effector(self):
        """找到末端执行器链接（通常为最远的叶子）"""
        leaf_links = []
        for link in self.links.keys():
            if link not in self.adj or not self.adj[link]:
                leaf_links.append(link)

        if leaf_links:
            max_depth = -1
            for link in leaf_links:
                depth = 0
                current = link
                while current != self.root_link:
                    parent_found = False
                    for parent, children in self.adj.items():
                        for jn, child in children:
                            if child == current:
                                current = parent
                                depth += 1
                                parent_found = True
                                break
                        if parent_found: break
                    if not parent_found: break
                if depth > max_depth:
                    max_depth = depth
                    self.end_effector_link = link
        else:
            self.end_effector_link = list(self.links.keys())[-1]
        print(f"末端执行器链接: {self.end_effector_link}")

    def compute_link_transforms(self, joint_positions=None):
        if joint_positions is None: joint_positions = JOINT_POS
        T = {self.root_link: np.eye(4)}
        stack=[self.root_link]; visited={self.root_link}
        c2j = {jd["child"]:(jn,jd) for jn,jd in self.joints.items()}
        while stack:
            pl=stack.pop()
            for (jn,cl) in self.adj.get(pl,[]):
                _, jd = c2j[cl]
                T_joint = T_from_xyz_rpy(jd["xyz"], jd["rpy"])
                q = float(joint_positions.get(jn, 0.0))
                if jd["type"] in ("revolute","continuous"):
                    T_var = rotation_about_axis(q, jd["axis"])
                elif jd["type"]=="prismatic":
                    T_var = translation_along_axis(q, jd["axis"])
                else:
                    T_var = np.eye(4)
                T[cl] = tf.concatenate_matrices(T[pl], T_joint, T_var)
                if cl not in visited: visited.add(cl); stack.append(cl)
        self.link_T = T
        return T

# =============== VTK 视图（一次建模，按需更新矩阵） ===============
class RobotViewer3D(QWidget):
    transformsUpdated = pyqtSignal()           # 机器人变换更新（供 UI 刷新姿态）
    sphereSelectionChanged = pyqtSignal(int)   # 选中球体索引变化（-1 表示无选中）
    sphereEdited = pyqtSignal()                # 球体属性被编辑（半径/位置变化）

    def __init__(self, robot: URDFRobot, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.renderer = vtk.vtkRenderer()
        self._set_dark_background()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        layout = QVBoxLayout(self); layout.addWidget(self.vtk_widget)

        self._build_lights()

        # === 坐标系（先创建，避免 update_transforms 时未初始化） ===
        self.base_axes_actor = self._make_axes_actor(length=0.15)
        self.ee_axes_actor   = self._make_axes_actor(length=0.15)
        self.base_axes_actor.SetVisibility(0)
        self.ee_axes_actor.SetVisibility(0)
        self.renderer.AddActor(self.base_axes_actor)
        self.renderer.AddActor(self.ee_axes_actor)

        # 路径绘制
        self.path_points = []
        self.path_actor = None
        self.show_path = False

        # 球体容器（每个元素：{"actor","source","axes","selected"}）
        self.spheres = []
        self.selected_sphere = -1

        # 机器人模型
        self.link_actors = {}
        self._build_actors_once()

        # 网格（初始隐藏）
        self.grid_actor = self._build_grid_actor(size=3.0, step=0.1, z=0.0)
        self.grid_actor.SetVisibility(0)
        self.renderer.AddActor(self.grid_actor)

        # 相机
        self._update_scene_bounds()
        self.reset_camera()
        self.bg_is_light = False

        # 拾取器与双击监听
        self.picker = vtk.vtkPropPicker()
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button_press, 1.0)

    # ---------- 背景 ----------
    def _set_dark_background(self):
        self.renderer.SetBackground(0.10,0.12,0.14)
        self.renderer.SetBackground2(0.05,0.05,0.06)
        self.renderer.SetGradientBackground(True)

    def _set_lightblue_background(self):
        self.renderer.SetGradientBackground(False)
        self.renderer.SetBackground(0.80, 0.88, 0.98)

    def toggle_background(self):
        self.bg_is_light = not self.bg_is_light
        if self.bg_is_light: self._set_lightblue_background()
        else: self._set_dark_background()
        self.vtk_widget.GetRenderWindow().Render()

    # ---------- 灯光 ----------
    def _build_lights(self):
        self.key_light = vtk.vtkLight()
        self.key_light.SetPosition(3,3,5); self.key_light.SetFocalPoint(0,0,0)
        self.key_light.SetColor(1,1,1); self.key_light.SetIntensity(0.9)
        self.renderer.AddLight(self.key_light)

        self.fill_light = vtk.vtkLight()
        self.fill_light.SetPosition(-3,-2,4); self.fill_light.SetFocalPoint(0,0,0)
        self.fill_light.SetColor(0.7,0.8,1.0); self.fill_light.SetIntensity(0.5)
        self.renderer.AddLight(self.fill_light)

    def set_key_intensity(self, value: float):
        self.key_light.SetIntensity(float(value))
        self.vtk_widget.GetRenderWindow().Render()

    def set_fill_intensity(self, value: float):
        self.fill_light.SetIntensity(float(value))
        self.vtk_widget.GetRenderWindow().Render()

    # ---------- 坐标系 ----------
    def _make_axes_actor(self, length=0.1):
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(length, length, length)
        axes.SetAxisLabels(False)
        axes.SetShaftTypeToCylinder()
        return axes

    def _np4x4_to_vtk(self, M: np.ndarray) -> vtk.vtkMatrix4x4:
        V = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                V.SetElement(i,j, float(M[i,j]))
        return V

    def _np4x4_to_vtkTransform(self, M: np.ndarray) -> vtk.vtkTransform:
        T = vtk.vtkTransform()
        T.SetMatrix(self._np4x4_to_vtk(M))
        return T

    def set_axes_length(self, length: float):
        self.base_axes_actor.SetTotalLength(length, length, length)
        self.ee_axes_actor.SetTotalLength(length, length, length)
        # 选中球体坐标系长度也同步
        for s in self.spheres:
            s["axes"].SetTotalLength(length*0.8, length*0.8, length*0.8)
        self.vtk_widget.GetRenderWindow().Render()

    def set_base_axes_visible(self, visible: bool):
        self.base_axes_actor.SetVisibility(1 if visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    def set_ee_axes_visible(self, visible: bool):
        self.ee_axes_actor.SetVisibility(1 if visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    def get_pose_xyz_rpy(self, link: str):
        if link in self.robot.link_T:
            T = self.robot.link_T[link]
        else:
            T = np.eye(4)
        xyz = T[:3, 3]
        rpy = tf.euler_from_matrix(T, axes='sxyz')
        return xyz, np.degrees(rpy)

    # ---------- 网格 ----------
    def _build_grid_actor(self, size=3.0, step=0.1, z=0.0) -> vtk.vtkActor:
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        half = size
        idx = 0
        for i in range(-int(half/step), int(half/step)+1):
            y = i*step
            pts.InsertNextPoint(-half, y, z); pts.InsertNextPoint(half, y, z)
            lines.InsertNextCell(2); lines.InsertCellPoint(idx); lines.InsertCellPoint(idx+1); idx += 2
        for i in range(-int(half/step), int(half/step)+1):
            x = i*step
            pts.InsertNextPoint(x, -half, z); pts.InsertNextPoint(x, half, z)
            lines.InsertNextCell(2); lines.InsertCellPoint(idx); lines.InsertCellPoint(idx+1); idx += 2

        poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.45,0.50,0.55); prop.SetOpacity(0.7); prop.SetLineWidth(1.0)
        return actor

    def set_grid_visible(self, visible: bool):
        self.grid_actor.SetVisibility(1 if visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    # ---------- 路径绘制 ----------
    def _create_path_actor(self):
        if self.path_actor:
            self.renderer.RemoveActor(self.path_actor)
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        if len(self.path_points) > 1:
            for i, point in enumerate(self.path_points):
                points.InsertNextPoint(point)
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(self.path_points))
            for i in range(len(self.path_points)):
                line.GetPointIds().SetId(i, i)
            lines.InsertNextCell(line)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points); poly_data.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly_data)
        self.path_actor = vtk.vtkActor(); self.path_actor.SetMapper(mapper)
        prop = self.path_actor.GetProperty()
        prop.SetColor(1.0, 0.0, 0.0); prop.SetLineWidth(3.0); prop.SetRenderLinesAsTubes(True)
        self.renderer.AddActor(self.path_actor)

    def add_path_point(self):
        if not self.show_path: return
        if self.robot.end_effector_link in self.robot.link_T:
            T = self.robot.link_T[self.robot.end_effector_link]
            position = T[:3, 3]
            if len(self.path_points) == 0 or np.linalg.norm(position - self.path_points[-1]) > 0.001:
                self.path_points.append(position.copy())
                self._create_path_actor()
                self.vtk_widget.GetRenderWindow().Render()

    def clear_path(self):
        self.path_points = []
        if self.path_actor:
            self.renderer.RemoveActor(self.path_actor)
            self.path_actor = None
        self.vtk_widget.GetRenderWindow().Render()

    def set_show_path(self, show: bool):
        self.show_path = show
        if not show:
            self.clear_path()
        elif show and len(self.path_points) > 1:
            self._create_path_actor()
            self.vtk_widget.GetRenderWindow().Render()

    # ---------- 球体 ----------
    def _new_sphere_entry(self, x, y, z, r, color=(0.0,1.0,0.0)):
        source = vtk.vtkSphereSource()
        source.SetCenter(x, y, z)
        source.SetRadius(r)
        source.SetPhiResolution(20); source.SetThetaResolution(20)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(*color); prop.SetAmbient(0.3); prop.SetDiffuse(0.7); prop.SetSpecular(0.4); prop.SetSpecularPower(20)

        # 该球体的局部坐标系（默认隐藏）
        axes = self._make_axes_actor(length=0.12)
        axes.SetVisibility(0)
        axes.SetPosition(x, y, z)

        self.renderer.AddActor(actor)
        self.renderer.AddActor(axes)

        return {"actor": actor, "source": source, "axes": axes, "selected": False}

    def add_sphere(self, x, y, z, radius, color=(0.0, 1.0, 0.0)):
        entry = self._new_sphere_entry(x, y, z, radius, color)
        self.spheres.append(entry)
        self.vtk_widget.GetRenderWindow().Render()
        return entry["actor"]

    def clear_all_spheres(self):
        for s in self.spheres:
            self.renderer.RemoveActor(s["actor"])
            self.renderer.RemoveActor(s["axes"])
        self.spheres = []
        self.selected_sphere = -1
        self.vtk_widget.GetRenderWindow().Render()
        self.sphereSelectionChanged.emit(-1)

    def _set_sphere_selected(self, idx: int):
        # 取消旧选中
        if 0 <= self.selected_sphere < len(self.spheres):
            old = self.spheres[self.selected_sphere]
            old["selected"] = False
            old["axes"].SetVisibility(0)
            old["actor"].GetProperty().SetEdgeVisibility(0)

        self.selected_sphere = idx

        # 高亮新选中
        if 0 <= idx < len(self.spheres):
            cur = self.spheres[idx]
            cur["selected"] = True
            cur["axes"].SetVisibility(1)
            cur["actor"].GetProperty().SetEdgeVisibility(1)
            cur["actor"].GetProperty().SetEdgeColor(1.0, 0.5, 0.0)

        self.vtk_widget.GetRenderWindow().Render()
        self.sphereSelectionChanged.emit(idx if 0 <= idx < len(self.spheres) else -1)

    def _on_left_button_press(self, obj, ev):
        # 检测双击
        iren = self.interactor
        repeat = iren.GetRepeatCount()  # 第二次点击时通常为 1
        x, y = iren.GetEventPosition()

        if repeat == 1:  # 视作双击
            if self.picker.Pick(x, y, 0, self.renderer):
                picked = self.picker.GetActor()
                # 查找是否命中某个球体
                for i, s in enumerate(self.spheres):
                    if s["actor"] is picked:
                        self._set_sphere_selected(i)
                        return  # 不把事件继续传给相机以免误操作
        # 仍然把事件交给默认交互样式
        iren.GetInteractorStyle().OnLeftButtonDown()

    def edit_selected_sphere(self, center=None, radius=None):
        """外部调用：按面板数据修改选中球体"""
        idx = self.selected_sphere
        if not (0 <= idx < len(self.spheres)): return
        s = self.spheres[idx]
        if center is not None:
            cx, cy, cz = center
            s["source"].SetCenter(float(cx), float(cy), float(cz))
            s["axes"].SetPosition(float(cx), float(cy), float(cz))
        if radius is not None:
            s["source"].SetRadius(float(radius))
        s["source"].Update()
        self.vtk_widget.GetRenderWindow().Render()
        self.sphereEdited.emit()

    def nudge_selected_sphere_along_axis(self, axis: str, step: float):
        """沿指定轴移动选中球体 (+/- step)"""
        idx = self.selected_sphere
        if not (0 <= idx < len(self.spheres)): return
        s = self.spheres[idx]
        cx, cy, cz = s["source"].GetCenter()
        if axis == 'X':
            cx += step
        elif axis == 'Y':
            cy += step
        elif axis == 'Z':
            cz += step
        self.edit_selected_sphere(center=(cx, cy, cz), radius=None)

    # ---------- 模型 ----------
    def _trimesh_to_polydata(self, mesh:trimesh.Trimesh) -> vtk.vtkPolyData:
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        verts = np.asarray(mesh.vertices, np.float64)
        if GLOBAL_SCALE!=1.0: verts = verts * GLOBAL_SCALE
        for v in verts: pts.InsertNextPoint(v.tolist())
        poly.SetPoints(pts)
        faces = np.asarray(mesh.faces, np.int64)
        cells = vtk.vtkCellArray()
        for f in faces:
            cells.InsertNextCell(3); cells.InsertCellPoint(int(f[0]))
            cells.InsertCellPoint(int(f[1])); cells.InsertCellPoint(int(f[2]))
        poly.SetPolys(cells)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly); normals.ComputeCellNormalsOn(); normals.ComputePointNormalsOn(); normals.Update()
        return normals.GetOutput()

    def _make_actor_from_poly(self, poly: vtk.vtkPolyData) -> vtk.vtkActor:
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetColor(0.78,0.80,0.88)
        prop.SetAmbient(0.18); prop.SetDiffuse(0.72)
        prop.SetSpecular(0.35); prop.SetSpecularPower(20)
        return actor

    def _build_actors_once(self):
        for lname, ldata in self.robot.links.items():
            for vis in ldata["visuals"]:
                path = resolve_mesh_path(vis["filename"])
                if not path:
                    print(f"[WARN] 缺失mesh: {lname} -> {vis['filename']}"); continue
                try:
                    mesh = trimesh.load_mesh(path, process=False)
                except Exception as e:
                    print(f"[WARN] 加载失败 {path}: {e}"); continue

                # 非均匀缩放烘焙进顶点（一次性）
                scale = np.asarray(vis["scale"], float)
                if not np.allclose(scale, [1,1,1]):
                    S = np.eye(4); S[0,0],S[1,1],S[2,2] = scale
                    mesh = mesh.copy(); mesh.apply_transform(S)

                poly = self._trimesh_to_polydata(mesh)
                actor = self._make_actor_from_poly(poly)
                self.renderer.AddActor(actor)

                T_visual = T_from_xyz_rpy(np.asarray(vis["xyz"],float), np.asarray(vis["rpy"],float)).astype(np.float64)
                self.link_actors.setdefault(lname, []).append({"actor":actor, "T_visual":T_visual})

        # 初始位姿
        self.robot.compute_link_transforms(JOINT_POS)
        self.update_transforms()

    def update_transforms(self):
        link_T = self.robot.link_T
        for lname, items in self.link_actors.items():
            T_link = link_T.get(lname, np.eye(4))
            for it in items:
                T_world = T_link.dot(it["T_visual"])
                it["actor"].SetUserMatrix(self._np4x4_to_vtk(T_world))

        # 末端路径点
        self.add_path_point()

        # 坐标系位姿同步
        self.base_axes_actor.SetUserTransform(self._np4x4_to_vtkTransform(np.eye(4)))
        if self.robot.end_effector_link in link_T:
            self.ee_axes_actor.SetUserTransform(self._np4x4_to_vtkTransform(link_T[self.robot.end_effector_link]))

        self.transformsUpdated.emit()
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self._update_scene_bounds()
        cam = self.renderer.GetActiveCamera()
        cx, cy, cz = self.scene_center
        dist = max(self.scene_radius*2.5, 0.5)
        cam.SetPosition(cx + dist*0.9, cy + dist*0.7, cz + dist*0.9)
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetViewUp(0,0,1)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    # ---------- 视图预设 ----------
    def _update_scene_bounds(self):
        bounds = [np.inf,-np.inf, np.inf,-np.inf, np.inf,-np.inf]
        has_any = False
        for items in self.link_actors.values():
            for it in items:
                bb = it["actor"].GetBounds()
                if bb is None: continue
                has_any = True
                bounds[0] = min(bounds[0], bb[0]); bounds[1] = max(bounds[1], bb[1])
                bounds[2] = min(bounds[2], bb[2]); bounds[3] = max(bounds[3], bb[3])
                bounds[4] = min(bounds[4], bb[4]); bounds[5] = max(bounds[5], bb[5])
        if not has_any:
            self.scene_center = np.array([0.0,0.0,0.0])
            self.scene_radius = 1.0
            return
        cx = 0.5*(bounds[0]+bounds[1]); cy = 0.5*(bounds[2]+bounds[3]); cz = 0.5*(bounds[4]+bounds[5])
        dx = bounds[1]-bounds[0]; dy = bounds[3]-bounds[2]; dz = bounds[5]-bounds[4]
        r = 0.5*max(dx,dy,dz)
        self.scene_center = np.array([cx,cy,cz], float)
        self.scene_radius = max(r, 1e-3)

    def set_view_preset(self, name: str):
        self._update_scene_bounds()
        c = self.scene_center; r = self.scene_radius
        cam = self.renderer.GetActiveCamera()

        if name == 'front':
            dir_vec = np.array([0.0, 1.0, 0.0]); up = np.array([0.0, 0.0, 1.0])
        elif name == 'side':
            dir_vec = np.array([1.0, 0.0, 0.0]); up = np.array([0.0, 0.0, 1.0])
        elif name == 'top':
            dir_vec = np.array([0.0, 0.0, 1.0]); up = np.array([0.0, 1.0, 0.0])
        else:  # 'iso'
            dir_vec = np.array([1.0, 1.0, 1.0]); dir_vec = dir_vec/np.linalg.norm(dir_vec); up = np.array([0.0, 0.0, 1.0])

        dist = max(r*5.2, 0.5)
        pos = c + dir_vec * dist
        cam.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        cam.SetFocalPoint(float(c[0]), float(c[1]), float(c[2]))
        cam.SetViewUp(float(up[0]), float(up[1]), float(up[2]))
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

# =============== 关节面板（紧凑版，双列布局） ===============
# =============== 关节面板（紧凑版，双列布局） ===============
class JointControlPanel(QWidget):
    def __init__(self, robot: URDFRobot, viewer: RobotViewer3D, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.viewer = viewer
        self.sliders = {}
        self.labels = {}  # 添加标签字典来保存标签引用

        lay = QVBoxLayout(self)
        lay.setSpacing(2)  # 减小间距
        title = QLabel("<b>关节控制</b>")
        lay.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        pane = QWidget()
        
        # 使用网格布局实现双列
        grid = QGridLayout(pane)
        grid.setSpacing(2)  # 减小间距
        grid.setContentsMargins(2, 2, 2, 2)  # 减小边距

        row = 0
        col = 0
        for jn in self.robot.joint_order:
            jd = self.robot.joints[jn]
            
            # 创建紧凑的组框
            g = QGroupBox(jn)
            g.setMaximumHeight(80)  # 限制高度
            g_l = QVBoxLayout(g)
            g_l.setSpacing(1)  # 极小间距
            g_l.setContentsMargins(3, 3, 3, 3)  # 减小内边距

            if jd["type"] in ("revolute","continuous"):
                lower, upper = jd["limit"]
                if jd["type"]=="continuous": lower, upper = -math.pi, math.pi
                if lower>=upper: lower, upper = -math.pi, math.pi
                s = QSlider(Qt.Horizontal)
                s.setRange(int(math.degrees(lower)), int(math.degrees(upper)))
                s.setValue(0)
                s.setMaximumHeight(20)  # 限制滑条高度
                lbl = QLabel("0°")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setMaximumHeight(15)  # 限制标签高度
                s.valueChanged.connect(self._mk_revolute_cb(jn, lbl))
                g_l.addWidget(lbl)
                g_l.addWidget(s)
                self.sliders[jn] = s
                self.labels[jn] = lbl  # 保存标签引用

            elif jd["type"]=="prismatic":
                lower, upper = jd["limit"]
                if not np.isfinite(lower) or not np.isfinite(upper) or lower>=upper: lower, upper = -0.2, 0.2
                s = QSlider(Qt.Horizontal)
                s.setRange(int(lower*1000), int(upper*1000))
                s.setValue(0)
                s.setMaximumHeight(20)
                lbl = QLabel("0 mm")
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setMaximumHeight(15)
                s.valueChanged.connect(self._mk_prismatic_cb(jn, lbl))
                g_l.addWidget(lbl)
                g_l.addWidget(s)
                self.sliders[jn] = s
                self.labels[jn] = lbl  # 保存标签引用
            else:
                lbl = QLabel("(fixed)")
                lbl.setMaximumHeight(15)
                g_l.addWidget(lbl)

            # 添加到网格
            grid.addWidget(g, row, col)
            col += 1
            if col >= 2:  # 双列
                col = 0
                row += 1

        scroll.setWidget(pane)
        lay.addWidget(scroll)

    def _mk_revolute_cb(self, jn, label):
        def cb(vdeg:int):
            label.setText(f"{vdeg}°")
            JOINT_POS[jn] = math.radians(vdeg)
            self.robot.compute_link_transforms(JOINT_POS)
            self.viewer.update_transforms()
        return cb

    def _mk_prismatic_cb(self, jn, label):
        def cb(vmm:int):
            label.setText(f"{vmm} mm")
            JOINT_POS[jn] = vmm/1000.0
            self.robot.compute_link_transforms(JOINT_POS)
            self.viewer.update_transforms()
        return cb

    def reset_all(self):
        for jn, s in self.sliders.items(): 
            s.blockSignals(True)
            s.setValue(0)
            s.blockSignals(False)
            
            # 手动更新标签
            if jn in self.labels:
                jd = self.robot.joints[jn]
                if jd["type"] in ("revolute", "continuous"):
                    self.labels[jn].setText("0°")
                elif jd["type"] == "prismatic":
                    self.labels[jn].setText("0 mm")
        
        JOINT_POS.clear()
        self.robot.compute_link_transforms(JOINT_POS)
        self.viewer.update_transforms()

# =============== 球体控制面板（紧凑版） ===============
class SphereControlPanel(QWidget):
    def __init__(self, viewer: RobotViewer3D, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        lay = QVBoxLayout(self)
        lay.setSpacing(2)
        title = QLabel("<b>球体控制</b>")
        lay.addWidget(title)

        # ----- 新建球体（使用网格布局） -----
        new_box = QGroupBox("新增球体")
        new_grid = QGridLayout(new_box)
        new_grid.setSpacing(2)
        new_grid.setContentsMargins(3, 3, 3, 3)
        
        # 坐标输入（2x2网格）
        self.nx = QDoubleSpinBox(); self.nx.setRange(-10,10); self.nx.setValue(0.0); self.nx.setSingleStep(0.1)
        self.ny = QDoubleSpinBox(); self.ny.setRange(-10,10); self.ny.setValue(0.0); self.ny.setSingleStep(0.1)
        self.nz = QDoubleSpinBox(); self.nz.setRange(-10,10); self.nz.setValue(0.0); self.nz.setSingleStep(0.1)
        self.nr = QDoubleSpinBox(); self.nr.setRange(0.01,2.0); self.nr.setValue(0.1); self.nr.setSingleStep(0.05)
        
        new_grid.addWidget(QLabel("X:"), 0, 0)
        new_grid.addWidget(self.nx, 0, 1)
        new_grid.addWidget(QLabel("Y:"), 0, 2)
        new_grid.addWidget(self.ny, 0, 3)
        new_grid.addWidget(QLabel("Z:"), 1, 0)
        new_grid.addWidget(self.nz, 1, 1)
        new_grid.addWidget(QLabel("R:"), 1, 2)
        new_grid.addWidget(self.nr, 1, 3)

        # 按钮
        self.btn_add = QPushButton("添加")
        self.btn_add.clicked.connect(lambda: self.viewer.add_sphere(self.nx.value(), self.ny.value(), self.nz.value(), self.nr.value()))
        self.btn_clear_all = QPushButton("清除")
        self.btn_clear_all.clicked.connect(self.viewer.clear_all_spheres)
        new_grid.addWidget(self.btn_add, 2, 0, 1, 2)
        new_grid.addWidget(self.btn_clear_all, 2, 2, 1, 2)
        
        lay.addWidget(new_box)

        # ----- 编辑选中球体（紧凑版） -----
        edit_box = QGroupBox("编辑选中")
        edit_grid = QGridLayout(edit_box)
        edit_grid.setSpacing(2)
        edit_grid.setContentsMargins(3, 3, 3, 3)
        
        self.sel_label = QLabel("未选中")
        edit_grid.addWidget(QLabel("当前:"), 0, 0)
        edit_grid.addWidget(self.sel_label, 0, 1, 1, 3)

        # 坐标编辑（2x2网格）
        self.cx = QDoubleSpinBox(); self.cx.setRange(-100,100); self.cx.setDecimals(3); self.cx.setSingleStep(0.01)
        self.cy = QDoubleSpinBox(); self.cy.setRange(-100,100); self.cy.setDecimals(3); self.cy.setSingleStep(0.01)
        self.cz = QDoubleSpinBox(); self.cz.setRange(-100,100); self.cz.setDecimals(3); self.cz.setSingleStep(0.01)
        self.cr = QDoubleSpinBox(); self.cr.setRange(0.001,10); self.cr.setDecimals(3); self.cr.setSingleStep(0.01)
        
        edit_grid.addWidget(QLabel("X:"), 1, 0)
        edit_grid.addWidget(self.cx, 1, 1)
        edit_grid.addWidget(QLabel("Y:"), 1, 2)
        edit_grid.addWidget(self.cy, 1, 3)
        edit_grid.addWidget(QLabel("Z:"), 2, 0)
        edit_grid.addWidget(self.cz, 2, 1)
        edit_grid.addWidget(QLabel("R:"), 2, 2)
        edit_grid.addWidget(self.cr, 2, 3)

        self.btn_apply = QPushButton("应用")
        self.btn_apply.clicked.connect(self._apply_edit)
        edit_grid.addWidget(self.btn_apply, 3, 0, 1, 4)

        # 移动控制（紧凑版）
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["X","Y","Z"])
        self.step = QDoubleSpinBox()
        self.step.setRange(0.001, 1.0)
        self.step.setDecimals(3)
        self.step.setValue(0.01)
        
        edit_grid.addWidget(QLabel("轴:"), 4, 0)
        edit_grid.addWidget(self.axis_combo, 4, 1)
        edit_grid.addWidget(QLabel("步:"), 4, 2)
        edit_grid.addWidget(self.step, 4, 3)

        self.btn_minus = QPushButton("-")
        self.btn_plus  = QPushButton("+")
        self.btn_minus.clicked.connect(lambda: self.viewer.nudge_selected_sphere_along_axis(self.axis_combo.currentText(), -self.step.value()))
        self.btn_plus.clicked.connect(lambda: self.viewer.nudge_selected_sphere_along_axis(self.axis_combo.currentText(),  self.step.value()))
        edit_grid.addWidget(self.btn_minus, 5, 0, 1, 2)
        edit_grid.addWidget(self.btn_plus, 5, 2, 1, 2)

        lay.addWidget(edit_box)

        # 事件连接
        self.viewer.sphereSelectionChanged.connect(self._refresh_from_selection)
        self.viewer.sphereEdited.connect(self._refresh_from_selection)

    def _apply_edit(self):
        self.viewer.edit_selected_sphere(center=(self.cx.value(), self.cy.value(), self.cz.value()),
                                         radius=self.cr.value())

    def _refresh_from_selection(self, *_):
        idx = self.viewer.selected_sphere
        if 0 <= idx < len(self.viewer.spheres):
            s = self.viewer.spheres[idx]
            cx, cy, cz = s["source"].GetCenter()
            r = s["source"].GetRadius()
            self.sel_label.setText(f"#{idx}")
            self.cx.setValue(cx)
            self.cy.setValue(cy)
            self.cz.setValue(cz)
            self.cr.setValue(r)
        else:
            self.sel_label.setText("未选中")

# =============== 显示控制面板（紧凑版） ===============
class DisplayControlPanel(QWidget):
    def __init__(self, viewer: RobotViewer3D, joint_panel: JointControlPanel, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.joint_panel = joint_panel
        
        lay = QVBoxLayout(self)
        lay.setSpacing(2)

        g = QGroupBox("显示设置")
        gl = QVBoxLayout(g)
        gl.setSpacing(2)
        gl.setContentsMargins(3, 3, 3, 3)

        # 第一行：背景和网格
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        self.btn_bg = QPushButton("切换背景")
        self.btn_bg.clicked.connect(self.viewer.toggle_background)
        self.chk_grid = QCheckBox("网格")
        self.chk_grid.stateChanged.connect(lambda s: self.viewer.set_grid_visible(s==Qt.Checked))
        row1.addWidget(self.btn_bg)
        row1.addWidget(self.chk_grid)
        gl.addLayout(row1)

        # 路径控制（一行）
        path_row = QHBoxLayout()
        path_row.setSpacing(2)
        self.chk_path = QCheckBox("路径")
        self.chk_path.stateChanged.connect(lambda s: self.viewer.set_show_path(s==Qt.Checked))
        self.btn_clear_path = QPushButton("清除路径")
        self.btn_clear_path.clicked.connect(self.viewer.clear_path)
        path_row.addWidget(self.chk_path)
        path_row.addWidget(self.btn_clear_path)
        gl.addLayout(path_row)

        # 光照控制（紧凑版）
        light_grid = QGridLayout()
        light_grid.setSpacing(1)
        light_grid.addWidget(QLabel("主光:"), 0, 0)
        self.s_key = QSlider(Qt.Horizontal)
        self.s_key.setRange(0, 200)
        self.s_key.setValue(90)
        self.s_key.setMaximumHeight(20)
        self.s_key.valueChanged.connect(lambda v: self.viewer.set_key_intensity(v/100.0))
        light_grid.addWidget(self.s_key, 0, 1)
        
        light_grid.addWidget(QLabel("辅光:"), 1, 0)
        self.s_fill = QSlider(Qt.Horizontal)
        self.s_fill.setRange(0, 200)
        self.s_fill.setValue(50)
        self.s_fill.setMaximumHeight(20)
        self.s_fill.valueChanged.connect(lambda v: self.viewer.set_fill_intensity(v/100.0))
        light_grid.addWidget(self.s_fill, 1, 1)
        gl.addLayout(light_grid)

        # 视图预设（一行四个按钮）
        view_row = QHBoxLayout()
        view_row.setSpacing(1)
        btn_front = QPushButton("正")
        btn_side  = QPushButton("侧")
        btn_top   = QPushButton("顶")
        btn_iso   = QPushButton("斜")
        for b in (btn_front, btn_side, btn_top, btn_iso):
            b.setMaximumHeight(25)
        btn_front.clicked.connect(lambda: self.viewer.set_view_preset('front'))
        btn_side.clicked.connect(lambda: self.viewer.set_view_preset('side'))
        btn_top.clicked.connect(lambda: self.viewer.set_view_preset('top'))
        btn_iso.clicked.connect(lambda: self.viewer.set_view_preset('iso'))
        view_row.addWidget(btn_front)
        view_row.addWidget(btn_side)
        view_row.addWidget(btn_top)
        view_row.addWidget(btn_iso)
        gl.addLayout(view_row)

        # 坐标系显示（紧凑版）
        axes_row1 = QHBoxLayout()
        axes_row1.setSpacing(2)
        self.chk_base_axes = QCheckBox("基座轴")
        self.chk_ee_axes   = QCheckBox("末端轴")
        self.chk_base_axes.stateChanged.connect(lambda s: self.viewer.set_base_axes_visible(s == Qt.Checked))
        self.chk_ee_axes.stateChanged.connect(lambda s: self.viewer.set_ee_axes_visible(s == Qt.Checked))
        axes_row1.addWidget(self.chk_base_axes)
        axes_row1.addWidget(self.chk_ee_axes)
        gl.addLayout(axes_row1)
        
        axes_row2 = QHBoxLayout()
        axes_row2.setSpacing(2)
        axes_row2.addWidget(QLabel("轴长:"))
        self.s_axes_len = QSlider(Qt.Horizontal)
        self.s_axes_len.setRange(5, 50)
        self.s_axes_len.setValue(15)
        self.s_axes_len.setMaximumHeight(20)
        self.s_axes_len.valueChanged.connect(lambda v: self.viewer.set_axes_length(v/100.0))
        axes_row2.addWidget(self.s_axes_len)
        gl.addLayout(axes_row2)

        # 回零按钮
        btn_reset = QPushButton("关节回零")
        btn_reset.clicked.connect(self.joint_panel.reset_all)
        gl.addWidget(btn_reset)

        lay.addWidget(g)
# =============== RRT路径规划面板 ===============
class RRTPathPlanningPanel(QWidget):
    def __init__(self, viewer: RobotViewer3D, robot: URDFRobot, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.robot = robot
        self.rrt_path = None
        self.smoothed_path = None
        self.path_actors = []  # VTK路径显示
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI界面"""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        
        # 标题
        title = QLabel("<b>RRT路径规划</b>")
        layout.addWidget(title)
        
        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        pane = QWidget()
        pane_layout = QVBoxLayout(pane)
        pane_layout.setSpacing(2)
        
        # 1. 路径起止点设置（紧凑版）
        points_group = QGroupBox("路径起止点")
        points_grid = QGridLayout(points_group)
        points_grid.setSpacing(2)
        points_grid.setContentsMargins(3, 3, 3, 3)
        
        # 起点输入
        points_grid.addWidget(QLabel("起点:"), 0, 0)
        self.start_x = QDoubleSpinBox()
        self.start_x.setRange(-1.0, 1.0)
        self.start_x.setDecimals(3)
        self.start_x.setValue(0.3)
        points_grid.addWidget(self.start_x, 0, 1)
        
        self.start_y = QDoubleSpinBox()
        self.start_y.setRange(-1.0, 1.0)
        self.start_y.setDecimals(3)
        self.start_y.setValue(0.0)
        points_grid.addWidget(self.start_y, 0, 2)
        
        self.start_z = QDoubleSpinBox()
        self.start_z.setRange(0, 1.0)
        self.start_z.setDecimals(3)
        self.start_z.setValue(0.3)
        points_grid.addWidget(self.start_z, 0, 3)
        
        # 终点输入
        points_grid.addWidget(QLabel("终点:"), 1, 0)
        self.end_x = QDoubleSpinBox()
        self.end_x.setRange(-1.0, 1.0)
        self.end_x.setDecimals(3)
        self.end_x.setValue(0.5)
        points_grid.addWidget(self.end_x, 1, 1)
        
        self.end_y = QDoubleSpinBox()
        self.end_y.setRange(-1.0, 1.0)
        self.end_y.setDecimals(3)
        self.end_y.setValue(0.2)
        points_grid.addWidget(self.end_y, 1, 2)
        
        self.end_z = QDoubleSpinBox()
        self.end_z.setRange(0, 1.0)
        self.end_z.setDecimals(3)
        self.end_z.setValue(0.4)
        points_grid.addWidget(self.end_z, 1, 3)
        
        # 使用当前末端位置按钮
        btn_use_current = QPushButton("使用当前末端位置作为起点")
        btn_use_current.clicked.connect(self.use_current_ee_as_start)
        points_grid.addWidget(btn_use_current, 2, 0, 1, 4)
        
        pane_layout.addWidget(points_group)
        
        # 2. RRT参数设置（紧凑版）
        params_group = QGroupBox("RRT参数")
        params_grid = QGridLayout(params_group)
        params_grid.setSpacing(2)
        params_grid.setContentsMargins(3, 3, 3, 3)
        
        # 最大迭代次数
        params_grid.addWidget(QLabel("迭代:"), 0, 0)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(100, 5000)
        self.max_iterations.setValue(1000)
        params_grid.addWidget(self.max_iterations, 0, 1)
        
        # 步长
        params_grid.addWidget(QLabel("步长:"), 0, 2)
        self.step_size = QDoubleSpinBox()
        self.step_size.setRange(0.01, 0.2)
        self.step_size.setDecimals(3)
        self.step_size.setValue(0.05)
        params_grid.addWidget(self.step_size, 0, 3)
        
        # 目标阈值
        params_grid.addWidget(QLabel("阈值:"), 1, 0)
        self.goal_threshold = QDoubleSpinBox()
        self.goal_threshold.setRange(0.01, 0.2)
        self.goal_threshold.setDecimals(3)
        self.goal_threshold.setValue(0.05)
        params_grid.addWidget(self.goal_threshold, 1, 1)
        
        # 平滑强度
        params_grid.addWidget(QLabel("平滑:"), 1, 2)
        self.smooth_strength = QSpinBox()
        self.smooth_strength.setRange(10, 100)
        self.smooth_strength.setValue(60)
        self.smooth_strength.setSuffix("%")
        params_grid.addWidget(self.smooth_strength, 1, 3)
        
        pane_layout.addWidget(params_group)
        
        # 3. 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(2)
        
        # 第一行按钮
        btn_row1 = QHBoxLayout()
        self.btn_plan = QPushButton("开始规划")
        self.btn_plan.clicked.connect(self.plan_path)
        self.btn_smooth = QPushButton("平滑路径")
        self.btn_smooth.clicked.connect(self.smooth_path)
        btn_row1.addWidget(self.btn_plan)
        btn_row1.addWidget(self.btn_smooth)
        control_layout.addLayout(btn_row1)
        
        # 第二行按钮
        btn_row2 = QHBoxLayout()
        self.btn_clear = QPushButton("清除路径")
        self.btn_clear.clicked.connect(self.clear_path)
        self.btn_execute = QPushButton("执行路径")
        self.btn_execute.clicked.connect(self.execute_path)
        self.btn_execute.setEnabled(False)
        btn_row2.addWidget(self.btn_clear)
        btn_row2.addWidget(self.btn_execute)
        control_layout.addLayout(btn_row2)
        
        # 状态显示
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("QLabel { color: lightgreen; }")
        control_layout.addWidget(self.status_label)
        
        pane_layout.addWidget(control_group)
        
        # 4. 路径信息
        info_group = QGroupBox("路径信息")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(2)
        
        self.info_labels = {}
        info_items = [
            ("路径点数:", "points", "0"),
            ("路径长度:", "length", "0.000 m"),
            ("障碍物数:", "obstacles", "0"),
            ("规划时间:", "time", "0.00 s")
        ]
        
        for i, (label, key, default) in enumerate(info_items):
            info_layout.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            value_label = QLabel(default)
            self.info_labels[key] = value_label
            info_layout.addWidget(value_label, i // 2, (i % 2) * 2 + 1)
        
        pane_layout.addWidget(info_group)
        
        pane_layout.addStretch()
        scroll.setWidget(pane)
        layout.addWidget(scroll)
    
    def use_current_ee_as_start(self):
        """使用当前末端执行器位置作为起点"""
        if self.robot.end_effector_link in self.robot.link_T:
            T = self.robot.link_T[self.robot.end_effector_link]
            pos = T[:3, 3]
            self.start_x.setValue(pos[0])
            self.start_y.setValue(pos[1])
            self.start_z.setValue(pos[2])
            self.status_label.setText("已设置当前位置为起点")
    
    def plan_path(self):
        """执行RRT路径规划"""
        import time
        start_time = time.time()
        
        # 获取起止点
        start = np.array([self.start_x.value(), self.start_y.value(), self.start_z.value()])
        goal = np.array([self.end_x.value(), self.end_y.value(), self.end_z.value()])
        
        # 获取障碍物（使用viewer中的球体）
        obstacles = self.viewer.spheres
        self.info_labels['obstacles'].setText(str(len(obstacles)))
        
        self.status_label.setText("正在规划路径...")
        QApplication.processEvents()
        
        # 执行RRT算法
        self.rrt_path = self.rrt_algorithm(start, goal, obstacles)
        
        planning_time = time.time() - start_time
        self.info_labels['time'].setText(f"{planning_time:.2f} s")
        
        if self.rrt_path is not None:
            self.status_label.setText("路径规划成功！")
            self.info_labels['points'].setText(str(len(self.rrt_path)))
            self.info_labels['length'].setText(f"{self.calculate_path_length(self.rrt_path):.3f} m")
            self.btn_smooth.setEnabled(True)
            self.btn_execute.setEnabled(True)
            self.visualize_path()
        else:
            self.status_label.setText("路径规划失败！")
            self.btn_smooth.setEnabled(False)
            self.btn_execute.setEnabled(False)
    
    def rrt_algorithm(self, start, goal, obstacles):
        """RRT算法实现"""
        max_iter = self.max_iterations.value()
        step_size = self.step_size.value()
        goal_threshold = self.goal_threshold.value()
        
        # 初始化树
        tree = {0: {'pos': start, 'parent': None}}
        
        for i in range(max_iter):
            # 随机采样（10%概率直接采样目标点）
            if np.random.random() < 0.1:
                random_point = goal
            else:
                # 在工作空间内随机采样
                random_point = np.array([
                    np.random.uniform(-0.8, 0.8),
                    np.random.uniform(-0.8, 0.8),
                    np.random.uniform(0, 0.8)
                ])
            
            # 找最近节点
            nearest_id = self.find_nearest_node(tree, random_point)
            nearest_pos = tree[nearest_id]['pos']
            
            # 计算新节点
            direction = random_point - nearest_pos
            distance = np.linalg.norm(direction)
            
            if distance > step_size:
                direction = direction / distance * step_size
            
            new_pos = nearest_pos + direction
            
            # 检查碰撞
            if not self.check_collision(nearest_pos, new_pos, obstacles):
                new_id = len(tree)
                tree[new_id] = {'pos': new_pos, 'parent': nearest_id}
                
                # 检查是否到达目标
                if np.linalg.norm(new_pos - goal) < goal_threshold:
                    # 回溯路径
                    path = []
                    current_id = new_id
                    while current_id is not None:
                        path.append(tree[current_id]['pos'])
                        current_id = tree[current_id]['parent']
                    path.reverse()
                    path.append(goal)
                    return np.array(path)
        
        return None
    
    def find_nearest_node(self, tree, point):
        """找到树中最近的节点"""
        min_dist = float('inf')
        nearest_id = 0
        
        for node_id, node in tree.items():
            dist = np.linalg.norm(node['pos'] - point)
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        
        return nearest_id
    
    def check_collision(self, start, end, obstacles):
        """检查路径段是否与障碍物碰撞"""
        # 离散化路径
        num_checks = 10
        for i in range(num_checks + 1):
            t = i / num_checks
            point = start * (1 - t) + end * t
            
            # 检查与每个球体的碰撞
            for sphere in obstacles:
                center = np.array(sphere['source'].GetCenter())
                radius = sphere['source'].GetRadius()
                if np.linalg.norm(point - center) < radius * 1.2:  # 留出安全边距
                    return True
        
        return False
    
    def smooth_path(self):
        """平滑路径"""
        if self.rrt_path is None:
            return
        
        self.status_label.setText("正在平滑路径...")
        QApplication.processEvents()
        
        strength = self.smooth_strength.value() / 100.0
        smoothed = [self.rrt_path[0]]
        
        i = 0
        while i < len(self.rrt_path) - 1:
            current = smoothed[-1]
            
            # 尝试跳过中间点
            max_skip = int(len(self.rrt_path) * strength)
            for j in range(min(i + max_skip, len(self.rrt_path) - 1), i, -1):
                if not self.check_collision(current, self.rrt_path[j], self.viewer.spheres):
                    smoothed.append(self.rrt_path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(self.rrt_path):
                    smoothed.append(self.rrt_path[i])
        
        self.smoothed_path = np.array(smoothed)
        self.info_labels['points'].setText(f"{len(self.smoothed_path)} (平滑后)")
        self.info_labels['length'].setText(f"{self.calculate_path_length(self.smoothed_path):.3f} m")
        self.status_label.setText("路径平滑完成！")
        self.visualize_path()
    
    def calculate_path_length(self, path):
        """计算路径长度"""
        if path is None or len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i+1] - path[i])
        return length
    
    def visualize_path(self):
        """在3D视图中显示路径"""
        # 清除之前的路径
        self.clear_path_visualization()
        
        # 创建路径线条
        if self.rrt_path is not None:
            self.create_path_actor(self.rrt_path, color=(1.0, 1.0, 0.0), width=2)  # 黄色原始路径
        
        if self.smoothed_path is not None:
            self.create_path_actor(self.smoothed_path, color=(1.0, 0.0, 1.0), width=3)  # 紫色平滑路径
        
        self.viewer.vtk_widget.GetRenderWindow().Render()
    
    def create_path_actor(self, path, color=(1,1,0), width=2):
        """创建路径的VTK Actor"""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        for point in path:
            points.InsertNextPoint(point)
        
        # 创建连续线条
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(path))
        for i in range(len(path)):
            polyline.GetPointIds().SetId(i, i)
        lines.InsertNextCell(polyline)
        
        # 创建PolyData
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        
        # 创建Mapper和Actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(width)
        actor.GetProperty().SetRenderLinesAsTubes(True)
        
        self.viewer.renderer.AddActor(actor)
        self.path_actors.append(actor)
        
        # 添加路径点标记
        for i, point in enumerate(path):
            if i == 0:  # 起点
                self.add_point_marker(point, color=(0,1,0), size=0.01)  # 绿色
            elif i == len(path) - 1:  # 终点
                self.add_point_marker(point, color=(1,0,0), size=0.01)  # 红色
    
    def add_point_marker(self, point, color=(1,0,0), size=0.01):
        """添加路径点标记"""
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(*point)
        sphere.SetRadius(size)
        sphere.SetPhiResolution(10)
        sphere.SetThetaResolution(10)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        
        self.viewer.renderer.AddActor(actor)
        self.path_actors.append(actor)
    
    def clear_path_visualization(self):
        """清除路径可视化"""
        for actor in self.path_actors:
            self.viewer.renderer.RemoveActor(actor)
        self.path_actors = []
    
    def clear_path(self):
        """清除路径"""
        self.rrt_path = None
        self.smoothed_path = None
        self.clear_path_visualization()
        self.viewer.vtk_widget.GetRenderWindow().Render()
        
        # 重置信息
        self.info_labels['points'].setText("0")
        self.info_labels['length'].setText("0.000 m")
        self.info_labels['time'].setText("0.00 s")
        self.status_label.setText("路径已清除")
        
        self.btn_smooth.setEnabled(False)
        self.btn_execute.setEnabled(False)
    
    def execute_path(self):
        """执行路径（沿路径移动机械臂）"""
        path = self.smoothed_path if self.smoothed_path is not None else self.rrt_path
        if path is None:
            return
        
        self.status_label.setText("执行路径中...")
        
        # 这里可以添加逆运动学求解和机械臂运动控制
        # 示例：简单的路径跟踪动画
        QMessageBox.information(self, "提示", 
                              f"路径包含{len(path)}个点\n"
                              f"总长度: {self.calculate_path_length(path):.3f}米\n"
                              f"实际执行需要逆运动学求解")
        
        self.status_label.setText("路径执行完成")
# =============== 主窗口 ===============
# class MainWindow(QMainWindow):
#     def __init__(self, robot: URDFRobot):
#         super().__init__()
#         self.setWindowTitle("URDF 机械臂（紧凑界面）")
#         self.resize(1200, 800)
        
#         cw = QWidget()
#         self.setCentralWidget(cw)
#         h = QHBoxLayout(cw)
#         h.setSpacing(2)

#         self.viewer = RobotViewer3D(robot)
#         h.addWidget(self.viewer, 5)

#         # 右侧面板
#         right = QWidget()
#         right.setMaximumWidth(320)  # 限制右侧面板宽度
#         vr = QVBoxLayout(right)
#         vr.setSpacing(2)
        
#         self.joint_panel = JointControlPanel(robot, self.viewer)
#         self.display_panel = DisplayControlPanel(self.viewer, self.joint_panel)
#         self.sphere_panel = SphereControlPanel(self.viewer)

#         vr.addWidget(self.display_panel)
#         vr.addWidget(self.sphere_panel)
#         vr.addWidget(self.joint_panel, 1)
        
#         h.addWidget(right, 2)
# =============== 主窗口（添加RRT面板） ===============
class MainWindow(QMainWindow):
    def __init__(self, robot: URDFRobot):
        super().__init__()
        self.setWindowTitle("URDF 机械臂（含RRT路径规划）")
        self.resize(1200, 800)
        
        cw = QWidget()
        self.setCentralWidget(cw)
        h = QHBoxLayout(cw)
        h.setSpacing(2)

        self.viewer = RobotViewer3D(robot)
        h.addWidget(self.viewer, 5)

        # 右侧面板使用选项卡
        right = QWidget()
        right.setMaximumWidth(320)
        vr = QVBoxLayout(right)
        vr.setSpacing(2)
        
        # 创建选项卡
        self.tabs = QTabWidget()
        
        # 控制选项卡
        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        control_layout.setSpacing(2)
        
        self.joint_panel = JointControlPanel(robot, self.viewer)
        self.display_panel = DisplayControlPanel(self.viewer, self.joint_panel)
        self.sphere_panel = SphereControlPanel(self.viewer)
        
        control_layout.addWidget(self.display_panel)
        control_layout.addWidget(self.sphere_panel)
        control_layout.addWidget(self.joint_panel, 1)
        
        self.tabs.addTab(control_tab, "控制")
        
        # RRT路径规划选项卡
        self.rrt_panel = RRTPathPlanningPanel(self.viewer, robot)
        self.tabs.addTab(self.rrt_panel, "路径规划")
        
        vr.addWidget(self.tabs)
        h.addWidget(right, 2)
# =============== 入口 ===============
def main():
    urdf = ensure_urdf_exists()
    print("URDF:", urdf)
    robot = URDFRobot(urdf)
    print(f"links={len(robot.links)}, joints={len(robot.joints)}, root='{robot.root_link}'")
    print("\n=== 机械臂树形结构 ===")
    for pre, _, node in RenderTree(robot.root_node):
        print(f"{pre}{node.name}")

    robot.compute_link_transforms(JOINT_POS)

    app = QApplication([])
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(53,53,53))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor(25,25,25))
    pal.setColor(QPalette.AlternateBase, QColor(53,53,53))
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, QColor(53,53,53))
    pal.setColor(QPalette.ButtonText, Qt.white)
    app.setPalette(pal)

    win = MainWindow(robot)
    win.show()
    print("提示：左键旋转，右键平移，滚轮缩放；双击球体选中并显示其坐标系")
    import sys
    sys.exit(app.exec_())

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
