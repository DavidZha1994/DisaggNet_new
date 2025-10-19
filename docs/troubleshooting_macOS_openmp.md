# macOS OpenMP 冲突（OMP Error #15）持久化修复指南

当同时加载 `libiomp5.dylib`（Intel/OpenMP，通常由 MKL 引入）和 `libomp.dylib`（LLVM/OpenMP，通常由 conda-forge 或 Homebrew 引入）时，会出现：

```
OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
```

以下是更持久、推荐的修复策略（按优先级）：

## 1) 使用 Conda 统一 OpenMP 实现（推荐）
- 选择一个栈，避免混用 MKL 与 conda-forge 的 OpenBLAS/LLVM-OpenMP。
- 方案 A：全套 conda-forge（OpenBLAS + LLVM OpenMP），避免 MKL。
  - 初始化环境：
    - `conda create -n disaggnet python=3.10 -y`
    - `conda activate disaggnet`
    - `conda config --add channels conda-forge`
  - 安装依赖（示例）：
    - `conda install pytorch torchvision cpuonly -c pytorch -y`
    - `conda install libomp numpy scipy scikit-learn -c conda-forge -y`
    - 若需 OpenBLAS 版本的 NumPy：`conda install "numpy=*=*openblas*" -c conda-forge -y`
- 方案 B：统一使用 MKL + Intel OpenMP。
  - `conda install mkl intel-openmp -y`
  - 确保移除 LLVM OpenMP：`conda remove libomp -y`

说明：不要在同一环境内混合安装 `mkl`/`intel-openmp` 与 `libomp`，保持 OpenMP 实现单一。

## 2) 使用 Homebrew 保持系统级 libomp 一致（可选）
- `brew install libomp`
- 将 Homebrew 的 `libomp` 放入动态库搜索路径（仅当你明确需要）：
  - `export DYLD_LIBRARY_PATH="$(brew --prefix)/opt/libomp/lib:$DYLD_LIBRARY_PATH"`

注意：一般不建议与 Conda 混用系统级 `DYLD_LIBRARY_PATH`，除非你清楚其影响。

## 3) 环境级兜底（快速但非根治）
- 在 shell 配置文件中加入：
  - `export KMP_DUPLICATE_LIB_OK=TRUE`
- 这是容忍重复初始化的兜底方案，能快速消除错误，但不从根源统一 OpenMP 实现。

## 4) 代码级兜底（已在 train.py 自动启用）
- 项目已在 `src/train.py` 顶部对 macOS 自动设置：
  - `KMP_DUPLICATE_LIB_OK=TRUE`
- 这可保证命令行直接运行项目时不再因 OMP 冲突崩溃，但依赖栈仍建议按 1) 统一。

## 5) 验证与排查
- 验证是否仍有冲突：运行一次训练或导入 `numpy`/`torch` 是否报 OMP #15。
- 若仍冲突，检查环境中是否同时存在：
  - `libiomp5.dylib`（来自 MKL / intel-openmp）
  - `libomp.dylib`（来自 LLVM OpenMP / conda-forge / Homebrew）
- 使用 `conda list | egrep "(mkl|openmp|libomp)"` 查看依赖，确保只保留一种 OpenMP 实现。

---

附：MPS/CUDA 精度与可视化
- 训练时可继续使用 BF16/FP16；项目已在序列可视化处强制将张量转换为 `float32`/`float64`，避免 BF16→NumPy 转换失败。
- 如需完全规避精度相关告警，可在配置中设置：`training.precision: 32`（即 FP32）。