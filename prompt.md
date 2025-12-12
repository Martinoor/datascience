# Tuning Cheat Sheet (churn-prediction-25-26)

- 入口：直接在 `model_construction.ipynb` 逐格执行。`SUBMIT_TO_KAGGLE` 控制是否自动提交；`run_note` 会自动带上 epoch/threshold。
- 日志：终格会写入 `tuning_log.csv`（超参 + val 指标 + Kaggle 分数 + 下一步建议）和 `submission_log.csv`（提交详情）。每次跑完先看 `tuning_log.csv` 是否 `improved_vs_prev=True`。
- 环境：保持 `prepare_datasets` 参数不变；需要 `~/.kaggle/kaggle.json`（chmod 600）和已安装 `kaggle` 包。
- 观测：看训练/验证 loss 走势、val AUC/F1/ACC、Kaggle 公共分数。阈值自动网格搜索，`best_threshold` 会随 val 表现更新。

## 调参逻辑速查
- 轻微过拟合（val loss 明显高于 train）：把 `dropout` 提到 0.18~0.22，`weight_decay` 到 2e-3，或 `max_seq_len` 再降到 400。
- 欠拟合（train/val 都高且贴近）：加容量（`num_layers` → 5 或 `dim_feedforward` → 896），或小幅增 `epochs` + `warmup_epochs`。
- 召回低、精度高：`pos_weight` *1.1~1.3，`focal_gamma` +0.2，并把阈值网格密集在 0.35~0.55。
- 精度低、召回高：`pos_weight` *0.8~0.9，`focal_gamma` -0.2，阈值上移 0.55~0.65。
- 已接近目标（Kaggle >0.65）：围绕当前配置小步扫描学习率（5e-4, 7e-4）、`max_seq_len`（400 vs 500）和阈值微调。

## 推荐接续实验队列
1) 先用当前 notebook 默认配置跑一遍，观察 `tuning_log.csv` 和曲线。
2) 若不过拟合且 val AUC < 0.66：试 `num_layers=5`, `dim_feedforward=896`, `dropout=0.18`, `lr=5e-4`。
3) 若过拟合：保持层数不变，`dropout=0.22`, `weight_decay=2e-3`, `max_seq_len=400`。
4) 若召回不足：在当前配置基础上将 `pos_weight` *1.2，`focal_gamma=1.8`，阈值网格改为 0.3~0.6 步长 0.01（在 val 阶段修改）。

## 使用提示
- 新配置务必更新 `run_note` 以便日志区分。