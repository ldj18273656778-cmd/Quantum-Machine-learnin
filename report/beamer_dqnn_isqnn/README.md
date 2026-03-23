# DQNN / ISQNN 周报 Beamer

## 文件说明

- `dqnn_isqnn_weekly_report.tex`: 汇报主文件（14 页）。
- `build_assets.py`: 生成代码结果图与摘要（不会改动原始 3 个脚本）。
- `assets/`: 公式截图、实验图表、数值摘要。

## 复现实验图表

在仓库根目录执行：

```powershell
& "C:\ProgramData\anaconda3\python.exe" "report\beamer_dqnn_isqnn\build_assets.py"
```

## 编译幻灯片

在 `report\beamer_dqnn_isqnn` 目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error "dqnn_isqnn_weekly_report.tex"
xelatex -interaction=nonstopmode -halt-on-error "dqnn_isqnn_weekly_report.tex"
```

输出文件：`dqnn_isqnn_weekly_report.pdf`
