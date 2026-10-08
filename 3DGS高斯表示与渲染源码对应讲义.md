# 3DGS 高斯表示与渲染：源码定位式中文讲义

> 对应仓库：`D:\02_学习科研\Git clone code\gaussian-splatting`  
> 对应提交：`ecaf58b4b50fad3a085fac4f2035e972b877b568`  
> 第一讲范围：高斯表示、COLMAP 初始化、相机、前向渲染  
> 暂不展开：损失函数、反向传播、参数更新、致密化、剪枝、CUDA kernel 细节

## 0. 先记住一条主线

这一讲只回答一个问题：

> COLMAP 中的一个三维点，怎样变成一个可学习的三维高斯，并最终影响渲染图像中的一个像素？

先不要把 3DGS 想成普通“点云”。一个高斯更像一个半透明、带颜色、可以旋转和拉伸的三维小椭球。许多椭球投影到屏幕上，再按照前后顺序叠加，就形成一张图像。

```text
COLMAP 稀疏点云与相机
        ↓
Scene 读取场景
        ↓
GaussianModel.create_from_pcd() 初始化 N 个高斯
        ↓
每个高斯保存：位置、尺度、旋转、透明度、球谐颜色
        ↓
从训练相机中选一个 viewpoint_camera
        ↓
gaussian_renderer.render() 整理相机与高斯参数
        ↓
GaussianRasterizer 将三维高斯投影成二维高斯并合成像素
        ↓
返回 render、radii、depth 等结果
```

本讲阅读代码时，所有命令都在仓库根目录执行：

```powershell
Set-Location 'D:\02_学习科研\Git clone code\gaussian-splatting'
```

---

## 1. 找到训练入口和总调用链

### 1.1 本步解决什么问题

先找到程序从哪里创建高斯、读取场景，又从哪里调用渲染器。现在只观察前向数据流，不分析损失和反向传播。

### 1.2 代码定位卡

```text
文件：train.py
目标：training() 和程序主入口
当前行号：training() 约第 43 行；主入口调用约第 282 行
搜索命令：rg -n "def training|GaussianModel\(|Scene\(|render_pkg = render|training\(" train.py
调用方：train.py 最底部的命令行入口
下一站：scene/__init__.py 中的 Scene.__init__()
```

### 1.3 关键代码

```python
# train.py:50-52
gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
scene = Scene(dataset, gaussians)
gaussians.training_setup(opt)
```

逐行看：

- `GaussianModel(...)`：先创建一个空的高斯模型容器。
- `Scene(dataset, gaussians)`：读取 COLMAP 场景，并把点云交给 `gaussians` 初始化。
- `training_setup(opt)`：为高斯参数建立优化器。这属于下一讲，本讲只知道这些参数稍后可以学习。

训练循环中，每次先选一台相机，再渲染：

```python
# train.py:97-103, 111-112
rand_idx = randint(0, len(viewpoint_indices) - 1)
viewpoint_cam = viewpoint_stack.pop(rand_idx)

render_pkg = render(viewpoint_cam, gaussians, pipe, bg, ...)
image = render_pkg["render"]
```

这里的两个核心输入是：

- `viewpoint_cam`：从哪里看、视野多大、输出图像多大；
- `gaussians`：场景中所有三维高斯的参数。

### 1.4 从哪里来，到哪里去

```text
命令行参数
  → training(...)
  → GaussianModel：承载全部高斯
  → Scene：读取点云和相机并初始化高斯
  → render(viewpoint_cam, gaussians, ...)：生成当前视角图像
```

### 1.5 现在先不要看

- `loss.backward()`：反向传播，下一讲再看；
- `optimizer.step()`：参数更新，下一讲再看；
- `densify_and_prune()`：高斯增删，下一讲再看；
- network GUI：不影响核心算法理解。

### 1.6 定位检查

请在 `train.py` 中找到 `render_pkg = render(...)`。确认传入的第二个参数是不是刚才创建并由 `Scene` 初始化的 `gaussians`。

---

## 2. 找到 COLMAP 数据如何进入高斯模型

### 2.1 本步解决什么问题

弄清楚你自己的 `data/sparse/0` 中的点云和相机，是怎样变成程序里的 `scene_info`，再进入高斯模型的。

### 2.2 代码定位卡

```text
文件：scene/__init__.py
目标：Scene.__init__()
当前行号：约第 25-83 行
搜索命令：rg -n "def __init__|sceneLoadTypeCallbacks|create_from_pcd" scene/__init__.py
调用方：train.py:51 的 Scene(dataset, gaussians)
下一站：scene/gaussian_model.py 的 create_from_pcd()
```

### 2.3 场景类型判断

```python
# scene/__init__.py:43-47
if os.path.exists(os.path.join(args.source_path, "sparse")):
    scene_info = sceneLoadTypeCallbacks["Colmap"](...)
elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
    scene_info = sceneLoadTypeCallbacks["Blender"](...)
```

你的数据目录中存在 `data/sparse`，所以进入 `Colmap` 分支。

`scene_info` 主要包含：

```python
# scene/dataset_readers.py:40-46
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool
```

点云最终来自：

```python
# scene/dataset_readers.py:120-126
def fetchPly(path):
    plydata = PlyData.read(path)
    positions = ...  # [N, 3]
    colors = ...     # [N, 3]，归一化到 0~1
    normals = ...    # [N, 3]
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
```

对应你项目中的实际文件是：

```text
data/sparse/0/points3D.ply
```

如果 `.ply` 不存在，代码会先从 `points3D.bin` 或 `points3D.txt` 转换，位置在 `scene/dataset_readers.py:205-216`。

### 2.4 真正触发高斯初始化的位置

```python
# scene/__init__.py:77-83
if self.loaded_iter:
    self.gaussians.load_ply(...)
else:
    self.gaussians.create_from_pcd(
        scene_info.point_cloud,
        scene_info.train_cameras,
        self.cameras_extent
    )
```

这里有两条路径：

- 加载已有训练结果：调用 `load_ply()`；
- 从头训练：调用 `create_from_pcd()`。

你执行新的训练时走第二条路径。

### 2.5 从哪里来，到哪里去

```text
data/sparse/0/points3D.ply
  → fetchPly()
  → SceneInfo.point_cloud
  → Scene.__init__()
  → GaussianModel.create_from_pcd()
```

### 2.6 定位检查

在 `scene/__init__.py` 中找到第 83 行附近的 `create_from_pcd()`，说出它接收的三个参数分别是什么。答案应包含：点云、训练相机信息、场景尺度。

---

## 3. 找到一个高斯保存的全部参数

### 3.1 本步解决什么问题

一个 COLMAP 点只有位置和颜色，而一个可渲染、可学习的三维高斯还需要大小、方向和透明度。本步把数学符号与真实 Python 变量一一对应。

### 3.2 代码定位卡

```text
文件：scene/gaussian_model.py
目标：GaussianModel.__init__()、属性 getter、create_from_pcd()
当前行号：约第 50-176 行
搜索命令：rg -n "self\._xyz|self\._features|self\._scaling|self\._rotation|self\._opacity|def create_from_pcd" scene/gaussian_model.py
调用方：Scene.__init__()
下一站：gaussian_renderer/__init__.py 的 render()
```

### 3.3 空模型中的六类参数

```python
# scene/gaussian_model.py:54-59
self._xyz = torch.empty(0)
self._features_dc = torch.empty(0)
self._features_rest = torch.empty(0)
self._scaling = torch.empty(0)
self._rotation = torch.empty(0)
self._opacity = torch.empty(0)
```

假设场景当前有 `N` 个高斯，最大球谐阶数是 `D`，则参数含义如下。

| 数学含义 | 源码存储 | 激活后/读取方式 | 典型形状 | 是否学习 | 传给渲染器 |
|---|---|---|---:|---|---|
| 中心位置 \(\mu\) | `_xyz` | `get_xyz`，不变换 | `[N,3]` | 是 | `means3D` |
| 颜色的 0 阶球谐 | `_features_dc` | `get_features_dc` | `[N,1,3]` | 是 | `dc` 或 `shs` |
| 颜色的高阶球谐 | `_features_rest` | `get_features_rest` | `[N,(D+1)^2-1,3]` | 是 | `shs` |
| 三轴尺度原始值 | `_scaling` | `exp(_scaling)` | `[N,3]` | 是 | `scales` |
| 旋转四元数原始值 | `_rotation` | 归一化四元数 | `[N,4]` | 是 | `rotations` |
| 透明度 logit | `_opacity` | `sigmoid(_opacity)` | `[N,1]` | 是 | `opacities` |

下划线变量通常是优化器直接修改的“原始参数”，getter 给出物理上真正使用的值：

```python
# scene/gaussian_model.py:39-47, 102-130
self.scaling_activation = torch.exp
self.opacity_activation = torch.sigmoid
self.rotation_activation = torch.nn.functional.normalize

def get_scaling(self):
    return self.scaling_activation(self._scaling)

def get_rotation(self):
    return self.rotation_activation(self._rotation)

def get_opacity(self):
    return self.opacity_activation(self._opacity)
```

为什么不直接保存最终值？

- `exp` 保证尺度始终大于 0；
- `sigmoid` 保证透明度始终在 0 到 1 之间；
- 四元数归一化保证它表示合法旋转。

### 3.4 初始化代码逐行对应

```python
# scene/gaussian_model.py:151-171
fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
features[:, :3, 0] = fused_color

dist2 = torch.clamp_min(distCUDA2(...), 0.0000001)
scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

rots = torch.zeros((N, 4), device="cuda")
rots[:, 0] = 1

opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), ...))

self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
self._features_dc = nn.Parameter(...)
self._features_rest = nn.Parameter(...)
self._scaling = nn.Parameter(scales.requires_grad_(True))
self._rotation = nn.Parameter(rots.requires_grad_(True))
self._opacity = nn.Parameter(opacities.requires_grad_(True))
```

逐项解释：

1. `fused_point_cloud`：COLMAP 点坐标直接成为初始高斯中心。
2. `RGB2SH`：把点云 RGB 转为 0 阶球谐系数；高阶项初始为 0。
3. `distCUDA2`：根据邻近点距离估计初始大小。稀疏区域的初始高斯通常更大。
4. 三个轴使用相同初始尺度，所以刚开始近似球形，之后可以学习成椭球。
5. 四元数 `[1,0,0,0]` 表示单位旋转，即初始时不旋转。
6. 实际初始透明度设为 `0.1`，但 `_opacity` 保存的是其反 sigmoid 值。
7. `nn.Parameter(...requires_grad_(True))` 表示这些量可通过训练被更新。

> 注意：`features` 创建时是 `[N,3,(D+1)^2]`，保存前做了转置，所以 `_features_dc` 最终是 `[N,1,3]`，`_features_rest` 是 `[N,(D+1)^2-1,3]`。

### 3.5 直觉小结

一个高斯可以写成：

```text
位置：我在哪里？
尺度：我沿三个局部轴分别有多大？
旋转：我的三个局部轴朝向哪里？
透明度：我最多能遮住后方多少？
球谐颜色：从当前观察方向看，我呈现什么颜色？
```

### 3.6 定位检查

在 `create_from_pcd()` 中找到 `rots[:, 0] = 1`。它说明初始旋转四元数是什么？再找到 `0.1`，确认它代表实际初始透明度，而不是 `_opacity` 中直接保存的数值。

---

## 4. 找到尺度和旋转如何形成三维椭球

### 4.1 本步解决什么问题

理解为什么代码只保存 3 个尺度和 4 个旋转数，却能描述一个任意朝向的三维椭球。

### 4.2 代码定位卡

```text
文件一：scene/gaussian_model.py
目标：build_covariance_from_scaling_rotation()
当前行号：约第 33-37 行
搜索：rg -n "build_covariance_from_scaling_rotation|get_covariance" scene/gaussian_model.py

文件二：utils/general_utils.py
目标：build_scaling_rotation()
当前行号：约第 101-110 行
搜索：rg -n "def build_scaling_rotation|L = R @ L" utils/general_utils.py
```

### 4.3 关键代码

```python
# utils/general_utils.py:101-110
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), ...)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L
```

这里先建立对角尺度矩阵：

\[
S=\begin{bmatrix}s_x&0&0\\0&s_y&0\\0&0&s_z\end{bmatrix}
\]

再由四元数得到旋转矩阵 `R`，形成：

\[
L=RS
\]

随后：

```python
# scene/gaussian_model.py:33-37
L = build_scaling_rotation(scaling_modifier * scaling, rotation)
actual_covariance = L @ L.transpose(1, 2)
symm = strip_symmetric(actual_covariance)
```

所以三维协方差为：

\[
\Sigma=LL^T=(RS)(RS)^T=RSS^TR^T
\]

`strip_symmetric()` 只是利用协方差矩阵对称，只保存 6 个独立元素，减少传输和计算。

### 4.4 单高斯数值例子

假设激活后的尺度为：

\[
s=(2,1,0.5)
\]

且没有旋转，即 `R=I`，那么：

\[
S=\operatorname{diag}(2,1,0.5),\qquad
\Sigma=SS^T=\operatorname{diag}(4,1,0.25)
\]

直觉上它沿 x 轴最长、y 轴居中、z 轴最短。若加入旋转，轴长不变，但整个椭球方向改变。

三维高斯的概念形式是：

\[
G(x)=\exp\left[-\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)\right]
\]

- `x`：空间中的某个位置；
- `μ`：`get_xyz` 返回的高斯中心；
- `Σ`：由 `get_scaling` 和旋转构造的协方差；
- 离中心越远，`G(x)` 越小。

### 4.5 一个重要分支

在 `gaussian_renderer/__init__.py:64-68`：

```python
if pipe.compute_cov3D_python:
    cov3D_precomp = pc.get_covariance(scaling_modifier)
else:
    scales = pc.get_scaling
    rotations = pc.get_rotation
```

两条路径表达的是同一个高斯：

- Python 预计算 `cov3D_precomp`；
- 或把 `scales + rotations` 交给 Rasterizer 计算。

两者必须二选一，不能同时提供。

### 4.6 定位检查

找到 `actual_covariance = L @ L.transpose(1, 2)`，回答：源码直接优化的是完整 `3×3` 协方差矩阵吗？答案是否定的；它优化的是尺度和旋转，再构造协方差。

---

## 5. 找到相机参数

### 5.1 本步解决什么问题

高斯位于世界坐标中。为了知道它落到屏幕哪个位置，需要先把世界坐标变到相机坐标，再投影到二维屏幕。

### 5.2 代码定位卡

```text
文件：scene/cameras.py
目标：Camera.__init__()
当前行号：约第 19-89 行
搜索命令：rg -n "class Camera|world_view_transform|projection_matrix|full_proj_transform|camera_center" scene/cameras.py
调用方：scene/__init__.py 中 cameraList_from_camInfos(...)
下一站：gaussian_renderer/__init__.py 的 GaussianRasterizationSettings
```

### 5.3 关键代码

```python
# scene/cameras.py:86-89
self.world_view_transform = torch.tensor(
    getWorld2View2(R, T, trans, scale)
).transpose(0, 1).cuda()

self.projection_matrix = getProjectionMatrix(
    znear=self.znear,
    zfar=self.zfar,
    fovX=self.FoVx,
    fovY=self.FoVy
).transpose(0, 1).cuda()

self.full_proj_transform = (
    self.world_view_transform.unsqueeze(0)
    .bmm(self.projection_matrix.unsqueeze(0))
).squeeze(0)

self.camera_center = self.world_view_transform.inverse()[3, :3]
```

可以先按下面的直觉理解：

```text
世界坐标中的高斯中心
  → world_view_transform：换到这台相机的坐标系
  → projection_matrix：透视投影并考虑视野、近远裁剪面
  → 屏幕位置
```

这些矩阵在代码中采用了与当前 Rasterizer 匹配的转置和乘法约定。初学阶段不要混用其他教材的行向量/列向量写法来机械比对。

`camera_center` 还有另一个用途：计算“从相机指向高斯”的观察方向，进而由球谐系数得到视角相关颜色。

### 5.4 参数去向

在 `gaussian_renderer/__init__.py:36-49`，相机属性进入：

```python
raster_settings = GaussianRasterizationSettings(
    image_height=int(viewpoint_camera.image_height),
    image_width=int(viewpoint_camera.image_width),
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    viewmatrix=viewpoint_camera.world_view_transform,
    projmatrix=viewpoint_camera.full_proj_transform,
    campos=viewpoint_camera.camera_center,
    ...
)
```

注意 `projmatrix` 实际传入的是已经组合好的 `full_proj_transform`。

### 5.5 定位检查

找到 `camera_center` 的计算，再到渲染器中找到它作为 `campos` 被传入的位置。它不是图像中心，而是相机在世界坐标中的位置。

---

## 6. 找到 Python 前向渲染主线

### 6.1 本步解决什么问题

把前面分散的高斯参数和相机参数汇总起来，看它们如何被送进 Rasterizer。

### 6.2 代码定位卡

```text
文件：gaussian_renderer/__init__.py
目标：render()
当前行号：约第 18-128 行
搜索命令：rg -n "def render|GaussianRasterizationSettings|means3D|opacity =|rasterizer\(|return out" gaussian_renderer/__init__.py
调用方：train.py:111
下一站：diff_gaussian_rasterization.GaussianRasterizer.forward()
```

### 6.3 第一步：建立一个屏幕空间占位张量

```python
# gaussian_renderer/__init__.py:25-28
screenspace_points = torch.zeros_like(
    pc.get_xyz, requires_grad=True, device="cuda"
) + 0
screenspace_points.retain_grad()
```

它的形状和三维中心一样是 `[N,3]`，代码中命名为 `means2D`。这里不是在 Python 中真正计算投影坐标，而是建立一个可保留梯度的占位张量，供后续训练统计屏幕空间位置梯度。本讲只需知道它被传给 Rasterizer。

### 6.4 第二步：打包相机和画布设置

`GaussianRasterizationSettings` 包含：输出高宽、视场角、背景色、相机矩阵、相机位置、当前球谐阶数等。它回答的是“用哪台相机，向多大的画布渲染”。

### 6.5 第三步：读取高斯几何参数

```python
# gaussian_renderer/__init__.py:54-68
means3D = pc.get_xyz
means2D = screenspace_points
opacity = pc.get_opacity

if pipe.compute_cov3D_python:
    cov3D_precomp = pc.get_covariance(scaling_modifier)
else:
    scales = pc.get_scaling
    rotations = pc.get_rotation
```

| 渲染参数 | 来源 | 含义 | 形状 |
|---|---|---|---:|
| `means3D` | `pc.get_xyz` | 世界坐标中的高斯中心 | `[N,3]` |
| `means2D` | `screenspace_points` | 屏幕空间梯度占位 | `[N,3]` |
| `opacity` | `pc.get_opacity` | sigmoid 后的不透明度 | `[N,1]` |
| `scales` | `pc.get_scaling` | exp 后三轴尺度 | `[N,3]` |
| `rotations` | `pc.get_rotation` | 归一化四元数 | `[N,4]` |
| `cov3D_precomp` | `pc.get_covariance()` | 预计算协方差的 6 个独立量 | `[N,6]` |

### 6.6 第四步：准备颜色

代码有三种颜色入口：

1. `override_color`：外部强制指定颜色；
2. `colors_precomp`：在 Python 中把球谐系数算成当前视角 RGB；
3. `shs`：把球谐系数交给 Rasterizer 计算 RGB。

Python 计算颜色的分支是：

```python
# gaussian_renderer/__init__.py:75-85
shs_view = pc.get_features.transpose(1, 2).view(-1, 3, K)
dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(N, 1)
dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
```

球谐的直觉是：同一个高斯可以从不同观察方向呈现略有不同的颜色。`dir_pp_normalized` 就是该观察方向。

### 6.7 第五步：调用 Rasterizer

```python
# gaussian_renderer/__init__.py:102-110
rendered_image, radii, depth_image = rasterizer(
    means3D=means3D,
    means2D=means2D,
    shs=shs,
    colors_precomp=colors_precomp,
    opacities=opacity,
    scales=scales,
    rotations=rotations,
    cov3D_precomp=cov3D_precomp
)
```

这一行就是 Python 世界与高性能光栅化实现之间最关键的边界。

### 6.8 从哪里来，到哪里去

```text
GaussianModel getters ─┐
                      ├→ gaussian_renderer.render()
Camera attributes ────┘          ↓
                           rasterizer(...)
                                 ↓
                   rendered_image, radii, depth_image
```

### 6.9 定位检查

在 `render()` 中分别找到 `means3D`、`opacity`、`scales` 和 `rotations` 的赋值，确认它们是否都来自 `pc`，也就是 `GaussianModel`。

---

## 7. 找到 Python 与 CUDA 的边界

### 7.1 本步解决什么问题

知道 Python 把哪些参数交给编译扩展，以及编译扩展返回什么。第一讲不进入 CUDA kernel。

### 7.2 代码定位卡

```text
文件：submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
目标：GaussianRasterizer.forward() 与 _RasterizeGaussians.forward()
当前行号：约第 44-90、158-207 行
搜索命令：rg -n "class _RasterizeGaussians|def forward|_C.rasterize_gaussians|class GaussianRasterizer" submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
调用方：gaussian_renderer/__init__.py 中的 rasterizer(...)
下一站：_C.rasterize_gaussians 编译扩展
```

### 7.3 参数合法性检查

```python
# 包装层约第 178-182 行
if (shs is None and colors_precomp is None) or (...都不为空...):
    raise Exception(...)

if ((scales is None or rotations is None) and cov3D_precomp is None) or (...):
    raise Exception(...)
```

这里强制两组“二选一”：

- 球谐系数 `shs` 或预计算颜色 `colors_precomp`；
- `scales + rotations` 或预计算协方差 `cov3D_precomp`。

### 7.4 真正进入扩展的位置

```python
# 包装层约第 59-84 行
args = (
    raster_settings.bg,
    means3D,
    colors_precomp,
    opacities,
    scales,
    rotations,
    ...
    raster_settings.viewmatrix,
    raster_settings.projmatrix,
    ...
    sh,
    raster_settings.campos,
    ...
)

num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = \
    _C.rasterize_gaussians(*args)
```

Python 能直接证明的是：参数被重排后传给 `_C.rasterize_gaussians`，并返回颜色、半径、逆深度和若干内部缓存。

### 7.5 CUDA 黑盒内部的概念步骤

下面是算法层面的概括，不是本讲对 CUDA 逐行执行的证明：

1. 判断高斯是否位于相机视锥内；
2. 将三维中心投影到屏幕；
3. 将三维协方差投影为二维椭圆足迹；
4. 找出二维高斯覆盖的屏幕 tile；
5. 按深度组织同一 tile 中的高斯；
6. 从前向后进行 Alpha 合成；
7. 输出颜色、屏幕半径和逆深度。

如果以后深入 CUDA，入口是 `_C.rasterize_gaussians`，但现在停在 Python 包装层即可。

### 7.6 定位检查

找到 `_C.rasterize_gaussians(*args)`，再找到紧随其后的返回语句。确认 Python 包装层最终向上返回的是 `color`、`radii` 和 `invdepths` 三项。

---

## 8. 找到渲染输出及其去向

### 8.1 本步解决什么问题

Rasterizer 返回结果后，外层 `render()` 如何命名和整理它们？训练代码又取走了什么？

### 8.2 代码定位卡

```text
文件：gaussian_renderer/__init__.py
目标：render() 的 out 字典
当前行号：约第 117-128 行
搜索命令：rg -n '"render"|"viewspace_points"|"visibility_filter"|"radii"|"depth"' gaussian_renderer/__init__.py
调用方：Rasterizer 返回 rendered_image、radii、depth_image
下一站：train.py 中 render_pkg[...] 的读取
```

### 8.3 关键代码

```python
# gaussian_renderer/__init__.py:117-128
rendered_image = rendered_image.clamp(0, 1)
out = {
    "render": rendered_image,
    "viewspace_points": screenspace_points,
    "visibility_filter": (radii > 0).nonzero(),
    "radii": radii,
    "depth": depth_image
}
return out
```

| 输出键 | 含义 | 典型形状 | 后续用途 |
|---|---|---:|---|
| `render` | 当前相机下的 RGB 图像 | `[3,H,W]` | 与真实图像比较 |
| `viewspace_points` | 屏幕空间占位张量 | `[N,3]` | 后续读取其梯度 |
| `visibility_filter` | `radii > 0` 的高斯索引 | `[M,1]` 左右 | 选择本视角可见高斯 |
| `radii` | 每个高斯的屏幕半径；不可见时通常为 0 | `[N]` | 可见性与致密化统计 |
| `depth` | 当前视角的逆深度结果 | 通常 `[1,H,W]` | 深度输出或约束 |

> 注意：当前代码的 `visibility_filter` 使用 `.nonzero()`，所以它是索引张量，不是长度为 `N` 的布尔 mask。变量名容易让初学者误会。

训练入口取走这些结果的位置：

```python
# train.py:111-112
render_pkg = render(...)
image = render_pkg["render"]
viewspace_point_tensor = render_pkg["viewspace_points"]
visibility_filter = render_pkg["visibility_filter"]
radii = render_pkg["radii"]
```

### 8.4 现在先不要看

本讲只认识输出，不继续追踪：

- `image` 如何计算 L1 与 SSIM；
- `viewspace_point_tensor.grad` 如何影响致密化；
- `radii` 如何影响高斯增删。

### 8.5 定位检查

在 `gaussian_renderer/__init__.py` 找到 `out`，再在 `train.py` 找到 `render_pkg`。确认两边的字典键是否完全对应。

---

## 9. 用“单高斯到单像素”串起全部代码

### 9.1 第一步：一个 COLMAP 点变成高斯

假设 COLMAP 点为：

```text
位置 p = (1, 0, 4)
颜色 rgb = (1, 0, 0)，即红色
邻点距离估计得到尺度 s = (0.2, 0.2, 0.2)
初始旋转 q = (1, 0, 0, 0)
初始实际透明度 opacity = 0.1
```

代码对应：

```text
p        → fused_point_cloud → self._xyz → pc.get_xyz → means3D
rgb      → RGB2SH            → _features_dc/_rest → shs 或 colors_precomp
s        → log(s)            → self._scaling → exp → scales
q        → self._rotation    → normalize → rotations
0.1      → inverse_sigmoid   → self._opacity → sigmoid → opacities
```

### 9.2 第二步：形成三维椭球

`scales + rotations` 描述协方差：

\[
\Sigma=RSS^TR^T
\]

在本例中三个尺度相同、没有旋转，所以初始形状近似球形。

### 9.3 第三步：相机观察高斯

相机矩阵将 `means3D=(1,0,4)` 变换到当前相机坐标，再投影到二维屏幕。三维协方差也会对应成屏幕上的二维椭圆足迹。

假设这个二维高斯在目标像素处的足迹权重为：

\[
w=0.8
\]

这个 `w` 表示该像素离二维高斯中心较近，但并非正好在中心。

### 9.4 第四步：计算该高斯在像素处的 Alpha

高斯自身实际透明度假设为：

\[
o=0.5
\]

那么它在该像素处的有效 Alpha 可直观写为：

\[
\alpha=o\cdot w=0.5\times0.8=0.4
\]

这里：

- `o` 对应 `opacities`；
- `w` 来自投影后的二维高斯足迹；
- `α` 是当前高斯对这个具体像素的有效不透明度。

若高斯颜色为纯红 `c=(1,0,0)`，背景为黑色，则单高斯输出：

\[
C=\alpha c+(1-\alpha)c_{bg}
  =0.4(1,0,0)+0.6(0,0,0)
  =(0.4,0,0)
\]

### 9.5 第五步：加入后方第二个高斯

假设后方蓝色高斯在该像素的有效 Alpha 为 `0.5`。从前向后合成：

\[
C=\sum_i T_i\alpha_i c_i,
\qquad
T_i=\prod_{j<i}(1-\alpha_j)
\]

前方红色高斯：

\[
T_1=1,\qquad T_1\alpha_1c_1=1\times0.4\times(1,0,0)
\]

后方蓝色高斯只能穿过前方剩余的 `0.6`：

\[
T_2=1-0.4=0.6
\]

\[
T_2\alpha_2c_2=0.6\times0.5\times(0,0,1)=(0,0,0.3)
\]

黑色背景下最终颜色：

\[
C=(0.4,0,0)+(0,0,0.3)=(0.4,0,0.3)
\]

剩余透射率是 `0.6×0.5=0.3`；如果背景不是黑色，还要加上 `0.3 c_bg`。

这解释了为什么深度顺序重要：交换两个高斯的前后位置，颜色贡献就会改变。

### 9.6 公式与代码最终对照

| 公式量 | 直觉 | Python 代码来源 |
|---|---|---|
| \(\mu\) | 三维中心 | `pc.get_xyz` → `means3D` |
| \(S\) | 三轴大小 | `pc.get_scaling` → `scales` |
| \(R\) | 椭球朝向 | `pc.get_rotation` → `rotations` |
| \(\Sigma\) | 完整三维形状 | `pc.get_covariance()` 或 Rasterizer 内构造 |
| \(o\) | 高斯自身透明度 | `pc.get_opacity` → `opacities` |
| \(c_i\) | 视角相关颜色 | `shs` 或 `colors_precomp` |
| \(w\) | 该像素处二维高斯权重 | Rasterizer 内部计算 |
| \(\alpha_i\) | 该像素处有效不透明度 | Rasterizer 内由透明度与足迹得到 |
| \(T_i\) | 到达第 i 个高斯前剩余的透射率 | Rasterizer 内前向合成 |
| \(C\) | 最终像素颜色 | `rendered_image` → `out["render"]` |

---

## 10. 你现在应该能独立走通的代码路线

不要背行号。请按下面顺序亲自使用搜索命令定位：

```powershell
# 1. 找训练入口、模型和场景
rg -n "def training|GaussianModel\(|Scene\(|render_pkg = render" train.py

# 2. 找 COLMAP 分支和高斯初始化调用
rg -n "sceneLoadTypeCallbacks|create_from_pcd" scene/__init__.py

# 3. 找六类高斯参数及其初始化
rg -n "self\._xyz|self\._features|self\._scaling|self\._rotation|self\._opacity|def create_from_pcd" scene/gaussian_model.py

# 4. 找协方差构造
rg -n "build_covariance_from_scaling_rotation|get_covariance" scene/gaussian_model.py
rg -n "def build_scaling_rotation|L = R @ L" utils/general_utils.py

# 5. 找相机矩阵
rg -n "world_view_transform|projection_matrix|full_proj_transform|camera_center" scene/cameras.py

# 6. 找渲染参数和输出
rg -n "def render|means3D|opacity =|rasterizer\(|return out" gaussian_renderer/__init__.py

# 7. 找 Python/CUDA 边界
rg -n "class _RasterizeGaussians|_C.rasterize_gaussians|class GaussianRasterizer" submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
```

完整链路应能复述为：

```text
train.py: training()
  → GaussianModel(...)
  → Scene(...)
      → 读取 COLMAP point_cloud 和 cameras
      → GaussianModel.create_from_pcd(...)
  → 随机选择 viewpoint_camera
  → gaussian_renderer.render(...)
      → 读取高斯属性和相机属性
      → GaussianRasterizer(...)
          → Python 包装层
          → _C.rasterize_gaussians(...)
      → 返回 render、viewspace_points、visibility_filter、radii、depth
```

---

## 11. 常见混淆

### 11.1 高斯不是一个没有体积的点

`_xyz` 只是中心。`_scaling + _rotation` 决定它覆盖多大的三维区域以及朝向。

### 11.2 `_scaling` 不是实际尺度

实际尺度是 `exp(_scaling)`。这样训练时无论原始参数为何值，实际尺度都保持为正。

### 11.3 `_opacity` 不是实际透明度

实际值是 `sigmoid(_opacity)`，范围是 0 到 1。

### 11.4 opacity 不是某个像素最终使用的 alpha

同一个高斯对不同像素的足迹权重不同，因此有效 Alpha 还取决于该像素离二维高斯中心多远。

### 11.5 球谐系数不等于固定 RGB

0 阶球谐提供基础颜色，高阶球谐允许颜色随观察方向变化。

### 11.6 `means2D` 不是 Python 已经算好的二维坐标

当前代码创建的是形状 `[N,3]` 的梯度占位张量，真正的投影由 Rasterizer 完成。

### 11.7 `visibility_filter` 在当前版本中不是布尔数组

代码使用 `(radii > 0).nonzero()`，得到的是可见高斯索引。

### 11.8 `depth` 在当前 Rasterizer 包装层中实际来自 `invdepths`

包装层把 `_C.rasterize_gaussians()` 返回的 `invdepths` 作为第三项向上返回，外层命名为 `depth_image`。阅读和使用时要留意这是逆深度语义。

---

## 12. 学完后的自测

不看前文，尝试回答：

1. `train.py` 在哪一行附近创建 `GaussianModel`，又在哪一行附近调用 `render()`？
2. 你的 `points3D.ply` 是如何到达 `create_from_pcd()` 的？
3. `_scaling` 为什么要经过 `exp`？
4. 初始四元数 `[1,0,0,0]` 表示什么？
5. 为什么源码不直接优化完整的 `3×3` 协方差矩阵？
6. `world_view_transform` 与 `full_proj_transform` 分别解决什么问题？
7. `means3D`、`scales`、`rotations`、`opacities` 分别来自哪个 getter？
8. `shs` 与 `colors_precomp` 为什么只能二选一？
9. 为什么后方高斯的颜色贡献要乘以前方剩余透射率？
10. `render()` 返回的五个键分别是什么？

如果能沿源码回答其中 8 个以上，就已经建立了“高斯表示—相机—渲染”的第一层完整联系。

下一讲再沿同一条真实调用链继续：

```text
rendered image
→ L1 + SSIM 损失
→ loss.backward()
→ 高斯参数梯度
→ optimizer.step()
→ 屏幕空间梯度统计
→ clone / split / prune
```
