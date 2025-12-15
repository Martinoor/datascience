# Transformer 模型特征与指标说明

## 一、问题与数据 & 目标概况

- 任务（以 Kaggle 数据为准）：利用用户在观测期内的行为序列，预测该用户是否会流失（是否访问页面 `Cancellation Confirmation`）。
- 样本定义：
  - 数据来自 `train.parquet` / `test.parquet`，每一行是一条用户行为事件。
  - 每个用户的行为序列长度不同（有的只有几条，有的达到一万多条）。
- 数据时间范围（从 parquet 统计信息核对）：
  - `train.parquet`：`2018-10-01` ~ `2018-11-20`
  - `test.parquet`：`2018-10-01` ~ `2018-11-20`
- 关于“10 天窗口 after 2018-11-20”的说明：
  - 项目描述里提到了 “after 2018-11-20 的 10 天窗口”，但当前发下来的 `train/test.parquet` 事件时间本身就截止到 `2018-11-20`，因此**在现有数据上无法直接构造**“2018-11-20 之后 10 天”的标签。
  - 实操上（也与 Kaggle 的数据形态一致：test 没有取消事件、train 有取消事件）：本项目把 churn 定义为“用户是否出现过 `Cancellation Confirmation`”，并据此训练/提交。
- 标签（churn）构造（基于当前 pipeline）：
  - 对 `train.parquet` 按 `userId` 聚合，只要该用户全历史序列中任意事件的 `page == "Cancellation Confirmation"`，则记为 `churn = 1`，否则为 `0`。
  - 对于最终发生取消的用户，`Cancellation Confirmation` 通常出现在该用户序列的最后一条或最后几条事件，这条信息不能直接用于特征（否则泄露标签）。
- 当前数据标签统计（train）：
  - 用户数：19,140
  - churn=1：4,271（约 22.3%）
  - churn=0：14,869（约 77.7%）
- 评估指标（Kaggle 官方）：Balanced Accuracy Score（平衡准确率）
  - `balanced_acc = (TPR + TNR) / 2`
- 预测输出（提交文件）：`id,target`，其中 `target` 为 0/1（二分类标签）。
  - 佐证：数据目录自带的 `churn-prediction-25-26/example_submission.csv` 也是 0/1 标签提交。

---

!!!!项目的唯一描述：
 Here's the kaggle link for the competition : https://www.kaggle.com/competitions/churn-prediction-25-26
The competition is to be performed in groups of two. You'll have a report of 4 pages to submit by december 14th, presenting the methods you tested and used. For the defense you'll get 8 minutes of presentations + 7 minutes of questions, including on question on the labs, that may involve writing a code snippet.
kaggle.com
Churn prediction 25/26
Predict churn prediction from streaming service logs
9:19
The goal of the competition is to predict whether or not some users (whose user ids are in the test file) will churn in the window of 10 days that follows the given observations (ie after "2018-11-20"). We consider that a user churns when they visit the page 'Cancellation Confirmation' （已编辑） 
9:21
This is not a trivial challenge ; to get a decent score, we strongly advise you to start working on it right away.

## 二、现有特征是否充分？（基于 `feature_pipeline.py`）

### 2.1 已覆盖的主要信息
当前特征（给 Transformer 的 per-event 数值特征 + 类别 embedding）总体覆盖面已经比较全，核心包括：

- 行为序列本身：`page_id` / `prev_page_id`（由 page embedding 学序列模式）
- 时间与节奏：`seconds_since_prev_event`、`hour_sin/cos`、`dow_sin/cos`
- Session 结构：session 内事件序号、session 时长、进度等
- 订阅状态与变化：`level`、升级/降级累计次数、距上次 level change 的时间/事件数
- 质量/异常：历史 404 比例（`error occur`）
- 内容消费与偏好集中度：distinct song/artist、top1/top3 占比与次数
- 跳过行为：skip flag、skip ratio（累计 + rolling window）
- 用户背景：device / metro / state（类别 embedding）

### 2.2 可能的缺口（建议增补方向）
如果要继续挖收益，建议优先考虑“与 churn 更直接相关、且不会引入泄露”的信息：

- 更强的“活跃度/衰退”信号：以天为粒度的活跃天数、最近 N 天 session 数、最近活跃距 cutoff 的天数（recency）
- 更强的“行为突变”信号：最近 N 次/最近 N 天内关键页面占比的变化率（例如 Help/Settings/Error/Logout 的变化）
- 页面集合扩展：当前只对 `KEY_PAGES` 做了 rolling/count 特征；可尝试把与 churn 更相关的 page 加入（例如 Thumbs Down、Roll Advert、Add to Playlist 等，需先统计其在数据中是否常见）

### 2.3 是否需要删减？（建议删减/加速方向）
“删减”我建议按两类看：一类是纯冗余计算（一定该删），另一类是可能有用但需要 ablation 验证。

- 明确的冗余计算（已处理）：旧版本 `feature_engineer()` 曾生成全量 page one-hot，但 `prepare_datasets()` 随后会把这些 one-hot 列全部 drop 掉（`drop_cols = set(page_categories)`）；这块属于纯浪费，已在代码里移除。
- 需要 ablation 的计算：`get_song_stats_fast()` 和部分 rolling 特征计算成本高；若你发现训练耗时或缓存占用过大，可以把它们作为“可开关的特征组”，用线上分数/本地 balanced accuracy 做去除对比再决定保留与否。

## 三、目标/验证集与 Kaggle 分数差异：我认为问题主要不在“目标偏差”，而在“指标未对齐”

你现在的本地打印主要是 `val_auc` + “按 F1 选阈值”，但 Kaggle 的评估指标是 **Balanced Accuracy**，因此出现“本地看起来不错、线上分数不一致”是非常典型的。

### 3.1 目标（label）是否符合 Kaggle？
- 从数据本身核对：`train/test` 的事件时间都截止到 `2018-11-20`；`test` 中 `Cancellation Confirmation` 为 0 行，而 `train` 中有 4,271 行（且对应 4,271 个用户）。
- 这意味着：Kaggle 的 churn 标签基本就是“该用户是否出现过 `Cancellation Confirmation`”，与你当前 `feature_pipeline.prepare_datasets()` 的 label 构造是一致的（目标本身大概率没有偏）。

### 3.2 为什么你的 validation 和 Kaggle 分数差很多？
主要是两点“对齐问题”叠加：

- 评估指标不一致：你本地看的是 AUC（阈值无关），但线上算的是 Balanced Accuracy（阈值相关）。
- 阈值选择目标不一致：你用 F1 选阈值，但线上看的是 Balanced Accuracy；同一组概率输出下，这两个指标最优阈值通常不同。

### 3.3 建议你用什么方式做本地验证（先不改代码的结论版）
- 本地验证请至少同时记录两组数：`val_balanced_accuracy`（用与你提交一致的 0/1 输出） + `val_auc`（辅助观察排序质量）。
- 阈值选择请以 `balanced_accuracy` 为目标（扫描阈值即可），不要用 F1 来决定最终提交阈值。
- 若你想进一步让验证更贴近线上分布：把 split 从“随机按用户划分”逐步换成“更时间一致/更接近 test 的设定”（例如用更靠近 `2018-11-20` 的 cutoff 或者按时间做 holdout），再比较线上分数是否更稳定。
