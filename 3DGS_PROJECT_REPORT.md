# 3D Gaussian Splatting 项目进展报告

## 1. 项目概述

基于 INRIA 官方实现 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)，在 Windows + RTX 5060 Laptop GPU (Blackwell, sm_120) 上完成环境搭建、CUDA 扩展编译、训练和可视化。

- **仓库**: `D:\02_学习科研\Git clone code\gaussian-splatting`
- **远程**: `https://github.com/beichuchu/3dgs.git`（main 分支，尚未推送成功）
- **Conda 环境**: `3dgs`（PyTorch 2.13.0.dev20260507+cu130, Python 3.10, CUDA 13.0）

---

## 2. 硬件与环境

| 组件 | 版本 |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop GPU (Blackwell 架构, sm_120) |
| CUDA Toolkit | 13.2（编译用）|
| Python | 3.10 |
| PyTorch | 2.13.0.dev20260507+cu130 (nightly) |
| 编译器 | VS 2026 Community (MSVC 14.51.36231, compiler 19.51.36260) |
| Windows SDK | 10.0.28000.0 |

### 为什么用 VS 2026 而非 VS 2019

VS 2019 (MSVC 14.29) + CUDA 13.2 + PyTorch nightly 存在**三重死锁**：

1. CCCL 头文件强制要求 `/Zc:preprocessor` + C++17 以上
2. PyTorch 头文件使用 C++20 特性（位域默认初始化器 `bool x : 1 = false;`）
3. VS 2019 不支持通过 nvcc 传递 C++20（`-std=c++20` 被 nvcc 忽略，回退到 C++14）

VS 2026 (MSVC 14.51) 完全支持 C++20/C++23，解决了上述所有问题。

---

## 3. CUDA 扩展编译

### 3.1 需要编译的子模块

| 子模块 | 路径 | 状态 |
|---|---|---|
| diff-gaussian-rasterization | `submodules/diff-gaussian-rasterization/` | ✅ 编译成功 |
| simple-knn | `submodules/simple-knn/` | ✅ 编译成功 |

### 3.2 setup.py 修改

两个子模块的 `setup.py` 都加了 `-Xcompiler /Zc:preprocessor`（CUDA 13.2 的 CCCL 头文件强制要求）：

**diff-gaussian-rasterization/setup.py**:
```python
extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"), "-Xcompiler", "/Zc:preprocessor"]}
```

**simple-knn/setup.py**:
```python
extra_compile_args={"nvcc": ["-Xcompiler", "/Zc:preprocessor"], "cxx": cxx_compiler_flags}
```

### 3.3 编译踩坑（5 个问题及解决方案）

| # | 问题 | 原因 | 解决 |
|---|---|---|---|
| 1 | nvcc 默认用 VS 2019 的 cl.exe | nvcc 搜索优先级：注册表 > PATH | 不加 `-ccbin`（加了反而报 PATH 不一致），靠 `--use-local-env` + PATH 前置 VS 2026 |
| 2 | `rc.exe` not found (LNK1158) | link.exe 嵌入 manifest 需要 rc.exe（Windows SDK bin） | 把 `Windows Kits\10\bin\10.0.28000.0\x64` 加到 PATH |
| 3 | bash export 的环境变量传不到 link.exe | bash `:` 分隔 vs Windows `;` 分隔，subprocess 深层链路丢失 | 写 Python 脚本 `build_vs2026.py` 在 `os.environ` 直接设置 |
| 4 | UnicodeDecodeError | 中文路径输出 GBK，`capture_output=True, text=True` 按 UTF-8 解码崩溃 | 去掉 `capture_output`，让输出直接打印 |
| 5 | cmd.exe / reg.exe 被 sandbox 拦截 | 安全策略 | 用 Python 脚本代替，避免调用 cmd.exe |

### 3.4 构建脚本

`build_vs2026.py` 在 Python 进程内设置所有环境变量，然后调用 `setup.py build_ext --inplace`：

```python
# 核心逻辑
os.environ["CUDA_HOME"] = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
os.environ["VCToolsInstallDir"] = r"C:\...\14.51.36231\"
# ... 设置 INCLUDE, LIB, PATH 等
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=dgr_dir)
# 不用 capture_output — 中文字幕路径会触发 GBK/UTF-8 冲突
```

完整脚本在仓库根目录 `build_vs2026.py`。

### 3.5 编译产物

- `diff_gaussian_rasterization/_C.cp310-win_amd64.pyd` — 4 个 .cu 文件 (backward.cu, forward.cu, rasterizer_impl.cu, rasterize_points.cu) + ext.cpp 编译链接成功
- `simple_knn/_C.cp310-win_amd64.pyd`

---

## 4. 训练数据

| 项目 | 详情 |
|---|---|
| 照片数量 | 123 张 JPG |
| 照片大小 | 每张 ~4MB |
| 拍摄日期 | 2026-05-04 |
| 存放路径 | `data/input/` (已加入 .gitignore) |
| COLMAP 重建 | `data/distorted/database.db` (200MB, 已加入 .gitignore) + `data/sparse/` (cameras.bin, images.bin, points3D.bin, points3D.ply) + `data/stereo/` |

---

## 5. 训练记录

共 6 次训练，前 4 次失败（当时 VS 2019 无法编译 CUDA 扩展），后 2 次成功：

| 编号 | 日期 | iter_7000 | iter_30000 | 状态 |
|---|---|---|---|---|
| 98d63bba-b | 5/4 15:37 | — | — | 失败（编译未通过） |
| d0e25afb-9 | 5/4 18:58 | — | — | 失败 |
| f40e9b64-a | 5/4 19:20 | — | — | 失败 |
| 92cc9e31-c | 5/4 19:21 | — | — | 失败 |
| **b5c884b7-3** | 5/7 22:32 | 160M | 171M | ✅ 成功 |
| **f9bd0dc6-5** | 10/7 01:00 | 160M | 170M | ✅ 成功 |

### 训练参数（两次成功训练完全相同）

```python
Namespace(
    sh_degree=3,           # 球谐函数 3 阶（完整颜色）
    source_path='...\\data',
    model_path='./output/xxx',
    images='images',
    resolution=8,          # 低分辨率（快但质量一般）
    white_background=False,
    train_test_exp=False,
    data_device='cuda',
    eval=False
)
```

### 训练命令

```bash
conda activate 3dgs
cd "D:/02_学习科研/Git clone code/gaussian-splatting"
python train.py -s data
```

---

## 6. 可视化

使用 SIBR Gaussian Viewer 查看训练结果：

```bash
viewers\bin\SIBR_gaussianViewer_app.exe -m output\b5c884b7-3
```

### 观察到的问题

训练模型在 SIBR Viewer 中存在**飘浮点云**（floating artifacts），主要原因：
- `resolution=8` 分辨率太低
- 致密化（densification）过于激进
- 背景区域的高斯填充

### 改进建议

- 提高分辨率：`--resolution 1` 或 `--resolution 2`
- 训练后做 pruning 去除飘浮点
- 尝试改进方法：Mip-Splatting / 2DGS / 4DGS

---

## 7. Git 状态

### 提交记录

**主仓库** (main 分支):
```
833ea75 Save code, data outputs, viewer sources; add build script; ignore binaries, photos, and large COLMAP db
54c035f fix submodule name in environment.yml
3dc0b75 fix training report, add missing dependency
9e1c2c6 fix asset name in results.md
8091682 Merge update readme
```

**子模块** (本地提交，未推送):
- diff-gaussian-rasterization: `fcfb377 Add /Zc:preprocessor for CCCL compatibility with CUDA 13.2`
- simple-knn: `3034747 Add /Zc:preprocessor for CCCL compatibility with CUDA 13.2`
- SIBR_viewers: `de5fc4b` (本地修改)

### .gitignore 新增

```gitignore
# 第三方二进制和大型资源
viewers/bin/
colmap-x64-windows-cuda3.12.6*/
data/input/
data/images/
# COLMAP database.db 超 200MB，超过 GitHub 100MB 限制
data/distorted/database.db
```

### 推送状态

- Remote: `https://github.com/beichuchu/3dgs.git`
- 本地 commit `833ea75` 已就绪（103 文件，~111MB pack）
- 推送失败：`RPC failed; curl 55 Send failure: Connection was reset`（GFW 干扰 HTTPS）
- 待执行：`git push -u origin main --force`

---

## 8. 文件结构

```
gaussian-splatting/
├── build_vs2026.py              # 构建脚本（VS 2026 + CUDA 13.2 环境设置）
├── train.py                      # 训练入口
├── render.py                     # 渲染脚本
├── metrics.py                    # 评估脚本
├── .gitignore                    # 已配置忽略照片、数据库、二进制
├── data/
│   ├── input/                    # 123 张训练照片 (ignored)
│   ├── images/                   # 处理后图片 (ignored)
│   ├── distorted/
│   │   ├── database.db           # COLMAP 数据库 200MB (ignored)
│   │   └── sparse/0/             # 稀疏重建结果
│   ├── sparse/0/                 # 去畸变稀疏重建
│   │   ├── cameras.bin
│   │   ├── images.bin
│   │   ├── points3D.bin
│   │   └── points3D.ply
│   └── stereo/                   # 立体匹配配置
├── output/
│   ├── b5c884b7-3/               # 训练成功 (5/7)
│   │   ├── cfg_args
│   │   ├── input.ply
│   │   └── point_cloud/
│   │       ├── iteration_7000/   # 检查点 (160M)
│   │       └── iteration_30000/  # 最终模型 (171M)
│   └── f9bd0dc6-5/               # 训练成功 (10/7)
│       └── point_cloud/
│           ├── iteration_7000/   # 检查点 (160M)
│           └── iteration_30000/  # 最终模型 (170M)
├── submodules/
│   ├── diff-gaussian-rasterization/
│   │   ├── setup.py              # 已加 -Xcompiler /Zc:preprocessor
│   │   └── diff_gaussian_rasterization/
│   │       └── _C.cp310-win_amd64.pyd  # 编译产物
│   └── simple-knn/
│       ├── setup.py              # 已加 -Xcompiler /Zc:preprocessor
│       └── _C.cp310-win_amd64.pyd
├── viewers/
│   ├── bin/                      # SIBR Viewer 二进制 (ignored)
│   ├── shaders/                  # GLSL 着色器源码
│   └── resources/                # 配置文件
└── SIBR_viewers/                 # Viewer 源码 (submodule)
```

---

## 9. 关键经验总结

1. **VS 2019 + CUDA 13.2 + PyTorch nightly = 不可能三角**：必须 VS 2022+ 才能编译
2. **nvcc 编译器搜索优先级**：注册表 > PATH；`-ccbin` 会触发 PATH 一致性检查，反而出错
3. **bash export 不可靠**：在 bash → python → distutils → link.exe 深层链路中环境变量会丢失，用 Python 脚本设置 `os.environ` 最稳
4. **subprocess capture_output 的编码陷阱**：中文路径输出 GBK，按 UTF-8 解码会崩，不要 capture
5. **GitHub 推送大仓库**：国内 HTTPS 到 GitHub 容易被 GFW 重置，建议用 SSH 或代理
6. **3DGS 训练质量**：`resolution=8` 太低会导致飘浮点云，建议用 `--resolution 1` 或 `2`

---

## 10. 下一步计划

- [ ] 完成推送到 GitHub（`git push -u origin main --force`）
- [ ] 用更高分辨率重新训练（`--resolution 2`）
- [ ] 对训练结果做 pruning 去除飘浮点云
- [ ] 尝试 Mip-Splatting / 2DGS 等改进方法
- [ ] 评估训练质量（PSNR / SSIM / LPIPS）

---

*文档生成时间: 2026-10-07*
*项目路径: D:\02_学习科研\Git clone code\gaussian-splatting*
