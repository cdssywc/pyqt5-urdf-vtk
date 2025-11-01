
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025, Chen XingYu. All rights reserved.
#
# License: Non-Commercial Use Only / 仅限非商业使用
# -----------------------------------------------------------------------------
# 本代码及其衍生作品仅允许用于个人学习、学术研究与教学等非商业场景。
# 严禁任何形式的商业使用，包括但不限于：出售、付费服务、SaaS/在线服务、
# 广告变现、集成到商业产品或用于商业咨询/竞赛/投标等。如需商业授权，请
# 先行获得版权所有者书面许可并签署授权协议。
#
# 允许的非商业使用条件：
# 1) 保留本版权与许可声明；2) 在衍生作品/发表物中进行署名（Lu Yaoheng）并
# 标明来源仓库；3) 不得移除或修改本段声明。
#
# 免责声明：本代码按“现状”提供，不含任何明示或默示担保。作者不对因使用本
# 代码产生的任何直接或间接损失承担责任。使用者需自行评估并承担风险。
#
# English Summary:
# This code is provided for personal, academic, and research purposes only.
# Any commercial use (sale, paid service, SaaS, ad-monetization, integration
# into commercial products, consultancy, competitions, bids, etc.) is strictly
# prohibited without prior written permission from the copyright holder.
# Keep this notice intact and provide proper attribution in derived works.
# Provided "as is" without warranty of any kind. Use at your own risk.
#
# Contact / 商务与授权联系: <cdssywc@163.com>
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
URDF 机械臂查看器（Qt5 + VTK，增强控件 + 视图预设 + 路径绘制 + 球体添加）
- 一次加载 STL -> VTK Actor；滑条仅更新变换矩阵（高性能）
- 显示设置：背景浅蓝切换、地面网格开关、主/辅光亮度调节、视图预设（正/侧/顶/斜）
- 路径绘制：可勾选显示末端运动路径，可清除路径
- 球体添加：可设置球体大小和位置，相对于世界坐标系（机械臂基坐标原点）
"""

import os, math, numpy as np, xml.etree.ElementTree as ET
from anytree import Node, RenderTree
import transformations as tf
import trimesh

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QSlider, QGroupBox, QScrollArea, QSizePolicy, QPushButton, QCheckBox, QToolButton, 
    QHBoxLayout as QHBox, QLineEdit, QDoubleSpinBox, QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt
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
        """找到末端执行器链接（通常是最后一个没有子链接的链接）"""
        # 找出所有没有子链接的链接
        leaf_links = []
        for link in self.links.keys():
            if link not in self.adj or not self.adj[link]:
                leaf_links.append(link)
                
        # 如果有多个叶子链接，选择距离根最远的那个
        if leaf_links:
            max_depth = -1
            for link in leaf_links:
                depth = 0
                current = link
                while current != self.root_link:
                    # 找到父链接
                    parent_found = False
                    for parent, children in self.adj.items():
                        for jn, child in children:
                            if child == current:
                                current = parent
                                depth += 1
                                parent_found = True
                                break
                        if parent_found:
                            break
                    if not parent_found:
                        break
                
                if depth > max_depth:
                    max_depth = depth
                    self.end_effector_link = link
        else:
            # 如果没有叶子链接，则使用最后一个链接
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
        
        # 路径绘制相关 - 在构建actors之前初始化
        self.path_points = []  # 存储路径点
        self.path_actor = None
        self.show_path = False  # 是否显示路径
        
        # 球体相关
        self.sphere_actors = []  # 存储所有球体actor
        
        self.link_actors = {}  # link -> list of {"actor":vtkActor, "T_visual":np.ndarray(4x4)}
        self._build_actors_once()

        # 网格（初始隐藏）
        self.grid_actor = self._build_grid_actor(size=3.0, step=0.1, z=0.0)
        self.grid_actor.SetVisibility(0)
        self.renderer.AddActor(self.grid_actor)
        
        # 记录包围盒中心/半径（用于视图预设）
        self._update_scene_bounds()

        self.reset_camera()
        self.bg_is_light = False  # 当前背景模式

    # ---------- 背景 ----------
    def _set_dark_background(self):
        self.renderer.SetBackground(0.10,0.12,0.14)
        self.renderer.SetBackground2(0.05,0.05,0.06)
        self.renderer.SetGradientBackground(True)

    def _set_lightblue_background(self):
        self.renderer.SetGradientBackground(False)
        self.renderer.SetBackground(0.80, 0.88, 0.98)  # 浅蓝

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

    # ---------- 网格 ----------
    def _build_grid_actor(self, size=3.0, step=0.1, z=0.0) -> vtk.vtkActor:
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        half = size
        # 平行 X
        idx = 0
        for i in range(-int(half/step), int(half/step)+1):
            y = i*step
            pts.InsertNextPoint(-half, y, z); pts.InsertNextPoint(half, y, z)
            lines.InsertNextCell(2); lines.InsertCellPoint(idx); lines.InsertCellPoint(idx+1); idx += 2
        # 平行 Y
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
        """创建路径actor"""
        if self.path_actor:
            self.renderer.RemoveActor(self.path_actor)
            
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        if len(self.path_points) > 1:
            # 添加点到VTK
            for i, point in enumerate(self.path_points):
                points.InsertNextPoint(point)
                
            # 创建线
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(self.path_points))
            for i in range(len(self.path_points)):
                line.GetPointIds().SetId(i, i)
            lines.InsertNextCell(line)
        
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        
        self.path_actor = vtk.vtkActor()
        self.path_actor.SetMapper(mapper)
        
        # 设置路径属性
        prop = self.path_actor.GetProperty()
        prop.SetColor(1.0, 0.0, 0.0)  # 红色
        prop.SetLineWidth(3.0)
        prop.SetRenderLinesAsTubes(True)  # 使线条更粗更平滑
        
        self.renderer.AddActor(self.path_actor)
        
    def add_path_point(self):
        """添加当前末端执行器位置到路径"""
        if not self.show_path:
            return
            
        # 获取末端执行器位置
        if self.robot.end_effector_link in self.robot.link_T:
            T = self.robot.link_T[self.robot.end_effector_link]
            position = T[:3, 3]  # 提取位置向量
            
            # 如果这是第一个点，或者与上一个点距离足够远，则添加
            if len(self.path_points) == 0 or np.linalg.norm(position - self.path_points[-1]) > 0.001:
                self.path_points.append(position)
                self._create_path_actor()
                self.vtk_widget.GetRenderWindow().Render()
    
    def clear_path(self):
        """清除路径"""
        self.path_points = []
        if self.path_actor:
            self.renderer.RemoveActor(self.path_actor)
            self.path_actor = None
        self.vtk_widget.GetRenderWindow().Render()
        
    def set_show_path(self, show: bool):
        """设置是否显示路径"""
        self.show_path = show
        if not show:
            self.clear_path()
        elif show and len(self.path_points) > 1:
            self._create_path_actor()
            self.vtk_widget.GetRenderWindow().Render()
            
    # ---------- 球体操作 ----------
    def add_sphere(self, x, y, z, radius, color=(0.0, 1.0, 0.0)):
        """添加球体到场景中"""
        # 创建球体源
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetCenter(x, y, z)
        sphere_source.SetRadius(radius)
        sphere_source.SetPhiResolution(20)  # 设置球体精度
        sphere_source.SetThetaResolution(20)
        
        # 创建映射器和演员
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere_source.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # 设置球体颜色和属性
        prop = actor.GetProperty()
        prop.SetColor(color[0], color[1], color[2])  # 默认绿色
        prop.SetAmbient(0.3)
        prop.SetDiffuse(0.7)
        prop.SetSpecular(0.4)
        prop.SetSpecularPower(20)
        
        # 添加到渲染器和球体列表
        self.renderer.AddActor(actor)
        self.sphere_actors.append(actor)
        
        # 刷新视图
        self.vtk_widget.GetRenderWindow().Render()
        
        return actor
    
    def clear_all_spheres(self):
        """清除所有球体"""
        for actor in self.sphere_actors:
            self.renderer.RemoveActor(actor)
        self.sphere_actors = []
        self.vtk_widget.GetRenderWindow().Render()

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
        prop.SetColor(0.78,0.80,0.88)     # 浅金属色
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

    def _np4x4_to_vtk(self, M: np.ndarray) -> vtk.vtkMatrix4x4:
        V = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                V.SetElement(i,j, float(M[i,j]))
        return V

    def update_transforms(self):
        link_T = self.robot.link_T
        for lname, items in self.link_actors.items():
            T_link = link_T.get(lname, np.eye(4))
            for it in items:
                T_world = T_link.dot(it["T_visual"])
                it["actor"].SetUserMatrix(self._np4x4_to_vtk(T_world))
                
        # 更新路径点
        self.add_path_point()
        
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
        # 仅基于机器人actors估算包围盒（忽略地面网格）
        bounds = [np.inf,-np.inf, np.inf,-np.inf, np.inf,-np.inf]
        has_any = False
        for items in self.link_actors.values():
            for it in items:
                bb = it["actor"].GetBounds()  # 已包含UserMatrix
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
        """
        name: 'front' 正视（沿 +Y 看向原点，Z 轴朝上）
              'side'  侧视（沿 +X 看向原点，Z 轴朝上）
              'top'   顶视（沿 +Z 看向原点，Y 轴朝上）
              'iso'   斜视（等轴测）
        """
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

        dist = max(r*5.2, 0.5)   # 距离放大，画面更"松"
        pos = c + dir_vec * dist
        cam.SetPosition(float(pos[0]), float(pos[1]), float(pos[2]))
        cam.SetFocalPoint(float(c[0]), float(c[1]), float(c[2]))
        cam.SetViewUp(float(up[0]), float(up[1]), float(up[2]))
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

# =============== 关节面板（区分旋转/直线） ===============
class JointControlPanel(QWidget):
    def __init__(self, robot: URDFRobot, viewer: RobotViewer3D, parent=None):
        super().__init__(parent)
        self.robot = robot
        self.viewer = viewer
        self.sliders = {}

        lay = QVBoxLayout(self)
        title = QLabel("<b>关节控制</b>"); lay.addWidget(title)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        pane = QWidget(); pane_lay = QVBoxLayout(pane)

        for jn in self.robot.joint_order:
            jd = self.robot.joints[jn]
            g = QGroupBox(jn); g_l = QVBoxLayout(g)

            if jd["type"] in ("revolute","continuous"):
                lower, upper = jd["limit"]
                if jd["type"]=="continuous": lower, upper = -math.pi, math.pi
                if lower>=upper: lower, upper = -math.pi, math.pi
                s = QSlider(Qt.Horizontal)
                s.setRange(int(math.degrees(lower)), int(math.degrees(upper))); s.setValue(0)
                lbl = QLabel("0°"); lbl.setAlignment(Qt.AlignCenter)
                s.valueChanged.connect(self._mk_revolute_cb(jn, lbl))
                g_l.addWidget(lbl); g_l.addWidget(s); self.sliders[jn]=s

            elif jd["type"]=="prismatic":
                lower, upper = jd["limit"]
                if not np.isfinite(lower) or not np.isfinite(upper) or lower>=upper: lower, upper = -0.2, 0.2
                s = QSlider(Qt.Horizontal)
                s.setRange(int(lower*1000), int(upper*1000)); s.setValue(0)
                lbl = QLabel("0 mm"); lbl.setAlignment(Qt.AlignCenter)
                s.valueChanged.connect(self._mk_prismatic_cb(jn, lbl))
                g_l.addWidget(lbl); g_l.addWidget(s); self.sliders[jn]=s
            else:
                g_l.addWidget(QLabel("(fixed)"))

            pane_lay.addWidget(g)

        pane_lay.addStretch(1); scroll.setWidget(pane); lay.addWidget(scroll)

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
        for s in self.sliders.values(): s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        JOINT_POS.clear()
        self.robot.compute_link_transforms(JOINT_POS); self.viewer.update_transforms()

# =============== 球体控制面板 ===============
class SphereControlPanel(QWidget):
    def __init__(self, viewer: RobotViewer3D, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        
        lay = QVBoxLayout(self)
        title = QLabel("<b>球体控制</b>"); lay.addWidget(title)
        
        # 球体参数输入
        param_box = QGroupBox("球体参数")
        param_layout = QFormLayout(param_box)
        
        # 位置输入
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-10.0, 10.0)
        self.spin_x.setValue(0.0)
        self.spin_x.setSingleStep(0.1)
        param_layout.addRow("X 位置:", self.spin_x)
        
        self.spin_y = QDoubleSpinBox()
        self.spin_y.setRange(-10.0, 10.0)
        self.spin_y.setValue(0.0)
        self.spin_y.setSingleStep(0.1)
        param_layout.addRow("Y 位置:", self.spin_y)
        
        self.spin_z = QDoubleSpinBox()
        self.spin_z.setRange(-10.0, 10.0)
        self.spin_z.setValue(0.0)
        self.spin_z.setSingleStep(0.1)
        param_layout.addRow("Z 位置:", self.spin_z)
        
        # 半径输入
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.01, 2.0)
        self.spin_radius.setValue(0.1)
        self.spin_radius.setSingleStep(0.05)
        param_layout.addRow("半径:", self.spin_radius)
        
        lay.addWidget(param_box)
        
        # 按钮
        btn_layout = QHBoxLayout()
        
        self.btn_add = QPushButton("添加球体")
        self.btn_add.clicked.connect(self.add_sphere)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_clear = QPushButton("清除所有球体")
        self.btn_clear.clicked.connect(self.viewer.clear_all_spheres)
        btn_layout.addWidget(self.btn_clear)
        
        lay.addLayout(btn_layout)
        lay.addStretch(1)
        
    def add_sphere(self):
        """添加球体到场景"""
        x = self.spin_x.value()
        y = self.spin_y.value()
        z = self.spin_z.value()
        radius = self.spin_radius.value()
        
        self.viewer.add_sphere(x, y, z, radius)

# =============== 显示控制面板（背景/网格/光照/视图） ===============
class DisplayControlPanel(QWidget):
    def __init__(self, viewer: RobotViewer3D, joint_panel: JointControlPanel, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.joint_panel = joint_panel
        lay = QVBoxLayout(self)

        g = QGroupBox("显示设置"); gl = QVBoxLayout(g)

        # 背景切换
        self.btn_bg = QPushButton("切换背景：浅蓝/深色")
        self.btn_bg.clicked.connect(self.viewer.toggle_background)
        gl.addWidget(self.btn_bg)

        # 地面网格
        self.chk_grid = QCheckBox("显示地面网格")
        self.chk_grid.stateChanged.connect(lambda s: self.viewer.set_grid_visible(s==Qt.Checked))
        gl.addWidget(self.chk_grid)
        
        # 路径显示
        path_box = QGroupBox("末端路径"); path_lay = QHBoxLayout(path_box)
        self.chk_path = QCheckBox("显示路径")
        self.chk_path.stateChanged.connect(lambda s: self.viewer.set_show_path(s==Qt.Checked))
        self.btn_clear_path = QPushButton("清除路径")
        self.btn_clear_path.clicked.connect(self.viewer.clear_path)
        path_lay.addWidget(self.chk_path)
        path_lay.addWidget(self.btn_clear_path)
        gl.addWidget(path_box)

        # 光照亮度
        gl.addWidget(QLabel("主光亮度"))
        self.s_key = QSlider(Qt.Horizontal); self.s_key.setRange(0, 200); self.s_key.setValue(int(0.9*100))
        self.s_key.valueChanged.connect(lambda v: self.viewer.set_key_intensity(v/100.0))
        gl.addWidget(self.s_key)

        gl.addWidget(QLabel("辅光亮度"))
        self.s_fill = QSlider(Qt.Horizontal); self.s_fill.setRange(0, 200); self.s_fill.setValue(int(0.5*100))
        self.s_fill.valueChanged.connect(lambda v: self.viewer.set_fill_intensity(v/100.0))
        gl.addWidget(self.s_fill)

        # 视图预设（正/侧/顶/斜）
        view_box = QGroupBox("视图预设"); view_lay = QHBox()
        btn_front = QToolButton(); btn_front.setText("正"); btn_front.setToolTip("正视 Front")
        btn_side  = QToolButton(); btn_side.setText("侧"); btn_side.setToolTip("侧视 Side")
        btn_top   = QToolButton(); btn_top.setText("顶"); btn_top.setToolTip("顶视 Top")
        btn_iso   = QToolButton(); btn_iso.setText("斜"); btn_iso.setToolTip("斜视 Iso")
        for b in (btn_front, btn_side, btn_top, btn_iso):
            b.setFixedSize(28, 28); b.setAutoRaise(True)
        btn_front.clicked.connect(lambda: self.viewer.set_view_preset('front'))
        btn_side .clicked.connect(lambda: self.viewer.set_view_preset('side'))
        btn_top  .clicked.connect(lambda: self.viewer.set_view_preset('top'))
        btn_iso  .clicked.connect(lambda: self.viewer.set_view_preset('iso'))
        view_lay.addWidget(btn_front); view_lay.addWidget(btn_side)
        view_lay.addWidget(btn_top); view_lay.addWidget(btn_iso)
        view_box.setLayout(view_lay)
        gl.addWidget(view_box)

        # 回零
        btn_reset = QPushButton("关节回零")
        btn_reset.clicked.connect(self.joint_panel.reset_all)
        gl.addWidget(btn_reset)

        gl.addStretch(1)
        lay.addWidget(g)

# =============== 主窗口 ===============
class MainWindow(QMainWindow):
    def __init__(self, robot: URDFRobot):
        super().__init__()
        self.setWindowTitle("URDF 机械臂（Qt5 + VTK，视图预设版 + 路径绘制 + 球体添加）"); self.resize(1280,840)
        cw = QWidget(); self.setCentralWidget(cw)
        h = QHBoxLayout(cw)

        self.viewer = RobotViewer3D(robot); h.addWidget(self.viewer, 5)
        # 右侧堆叠：显示面板 + 关节面板 + 球体面板
        right = QWidget(); vr = QVBoxLayout(right)
        self.joint_panel = JointControlPanel(robot, self.viewer)
        self.display_panel = DisplayControlPanel(self.viewer, self.joint_panel)
        self.sphere_panel = SphereControlPanel(self.viewer)
        
        vr.addWidget(self.display_panel)
        vr.addWidget(self.sphere_panel)
        vr.addWidget(self.joint_panel, 1)  # 关节面板占剩余
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

    win = MainWindow(robot); win.show()
    print("提示：左键旋转，右键平移，滚轮缩放；右侧可切换背景/网格、调光、切换视图；滑条控制关节。")
    print("提示：勾选'显示路径'可在移动机械臂时绘制末端轨迹，点击'清除路径'可清除轨迹。")
    print("提示：使用球体控制面板可以添加和清除球体，球体位置相对于世界坐标系原点。")
    import sys; sys.exit(app.exec_())

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()