# 日频特征改造方案（user × day）

目标：把当前 **事件级（user × event）** 的特征/序列，改造成 **日频（user × date）** 的特征/序列，在不依赖文档假设的前提下基于真实数据可行性做设计。

---

## 1. 基于数据实测的结论（直接读 parquet）

数据文件：`churn-prediction-25-26/train.parquet`、`churn-prediction-25-26/test.parquet`

- 行为数据粒度：每行一条事件（含 `userId`、`page`、`time` 等）。
- 时间范围（train/test 一致）：`2018-10-01` ~ `2018-11-20`，共 **51 个自然日**（`time` 为 `datetime64`）。
- 规模：
  - train：`17,499,636` 行，`19,140` 个用户
  - test：`4,393,179` 行，`2,904` 个用户
- `page` 类别数较少且稳定：train 19 类、test 20 类、并集 22 类。
  - train 独有：`Cancel`、`Cancellation Confirmation`
  - test 独有：`Login`、`Register`、`Submit Registration`
  - 其余 **17 类为共有页面**（如 `NextSong`、`Thumbs Up/Down`、`Help`、`Settings`、`Upgrade/Downgrade` 等）
- 缺失情况：`song/artist/length` 约 **18.3%** 缺失（主要来自非 `NextSong` 事件）。
- 用户日活跃度（train）：
  - 用户平均活跃天数 `~11` 天（中位数 8 天，P90 24 天）
  - user-day 平均事件数 `~85`（中位数 58，P90 202）
- 取消（churn）时序（train）：
  - `Cancellation Confirmation` 出现的日期 **总是该用户最后活跃日期**（`churn_date == last_date`）
  - `Cancel` 在约 97% 的用户中发生在 `Cancellation Confirmation` 之前（同一天内更常见）

结论：**天然支持日频聚合**。全局只有 51 天，日频序列长度上界 51（远小于事件级 400~450），而且大多数用户只有 ~10 天左右的有效日序列，计算上更友好。

---

## 2. 现状梳理（你当前的实现）

你目前的训练入口（`model_construction.ipynb`）是：

- `feature_pipeline.prepare_datasets()`：对 train/test 做一致特征工程（事件级），并按用户划分 train/val
- `transformer_model.build_user_sequences()`：把事件表变成每用户一个序列（按事件顺序），截断到 `max_seq_len`
- Transformer 输入 token = “事件”，每个 token 含：
  - 数值特征：如 `length`、`error occur`、滚动的歌曲/歌手统计、skip/upgrade 等累计或 rolling 指标
  - 类别 embedding：`page_id`、`prev_page_id`、`metro_id/state_id/device_id`
- 防泄露：训练时会把 `Cancel`、`Cancellation Confirmation` 行删掉，并对每个用户裁掉末尾一段事件（`truncate_user_histories`）

这套方案的问题不是“不可用”，而是：

- 事件级序列太长且噪声大（单用户上千事件是常态），训练成本高
- 很多信号其实是 **日尺度** 的（活跃度、是否在折腾设置/帮助/套餐、听歌数量的趋势等）

---

## 3. 日频化的核心定义（建议先统一口径）

### 3.1 日（date）的定义

- 用 `time.dt.floor('D')` 作为自然日（UTC/本地时区先按数据原样，不额外改时区）
- 记为 `date`

### 3.2 日频样本的两种形态

**形态 A：user-day 表（长表）**

- 每行：`(userId, date)` 聚合后的特征
- 适合：做传统模型（LightGBM/XGBoost/LogReg），或者后续再按窗口抽取用户级特征

**形态 B：user 的“日序列”（用于 Transformer/RNN）**

- 每用户一条序列：按 `date` 排序的 daily feature 向量序列
- `seq_len` ≤ 51（固定长度也可）

### 3.3 缺失日期（没发生任何事件）的处理

这是日频方案里最关键的选择之一：

- 方案 1（推荐先做）：**只保留活跃日**（当日有事件才有 token）
  - 优点：序列更短；实现简单
  - 缺点：丢失“连续不活跃”的信息，需要额外加 `days_since_prev_active` 特征补回
- 方案 2（更贴近“日历”）：补齐全局日历（例如 51 天）并用 0 填充
  - 优点：模型可直接学习“尾部断崖式不活跃”
  - 风险：对“刚注册/刚出现”的用户，会出现大量前置 0，需要加 `has_started`/`days_since_first_seen` 区分

建议：先做方案 1 跑通；如果验证集上“尾部不活跃”很重要，再做方案 2。

---

## 4. 日频特征设计（v0：尽量用稳定、可解释、可泛化的聚合）

下面以 **一天内** 的聚合为主（`groupby(['userId','date'])`），再加少量跨日衍生（rolling/cumsum）。

### 4.1 活跃度与内容消费（核心）

- `events_total`：当天事件总数
- `nextsong_cnt`：当天 `page == 'NextSong'` 的次数
- `songs_unique` / `artists_unique`：当天 unique song/artist 数（仅 NextSong 有意义）
- `song_length_sum` / `song_length_mean` / `song_length_p50`：当天 `length` 聚合（只在 NextSong 非空）
- `page_unique`：当天 unique page 数（探索/使用复杂度）

可选跨日衍生（按用户对 daily 序列做）：

- `events_7d_mean` / `nextsong_7d_mean`：7 日 rolling 均值（趋势）
- `events_7d_slope`：近 7 天线性趋势（活跃上升/下降）

### 4.2 页面行为分布（更贴近你现在的 KEY_PAGES 思路）

由于 page 总类目不多，日频可以直接做“页面计数/占比”：

- 对共有的 17 个 page：做 `page_<name>_cnt`（当天计数）
- 再做比例：`page_<name>_ratio = page_<name>_cnt / events_total`（或除以 `nextsong_cnt`）

注意点（train/test page 集不完全一致）：

- train 独有 `Cancel` / `Cancellation Confirmation`：应 **从特征构造中剔除**（避免显式泄露）
- test 独有 `Login` / `Register` / `Submit Registration`：
  - 如果保留计数列：train 上这些列恒为 0，模型无法学习其意义（容易引入分布漂移）
  - 更稳的做法：把它们并入一个 `preauth_pages_cnt`（对 test 有值、train 基本为 0，但语义更明确）

### 4.3 订阅/套餐（level）相关

事件里 `level` 为 `free/paid`，日频建议同时保留“比例”和“末值”：

- `paid_ratio`：当天 paid 事件占比
- `paid_last`：当天最后一个事件的 level（free/paid）
- `upgrade_cnt` / `downgrade_cnt` / `submit_upgrade_cnt` / `submit_downgrade_cnt`（来自 page 计数）

跨日特征（可选）：

- `paid_change_flag`：当天 `paid_last` 是否与前一活跃日不同
- `days_since_last_level_change`

### 4.4 会话（session）相关（建议日频一定加，信息密度高）

你原 pipeline 丢了 `sessionId`，但日频聚合里它非常有用：

- `sessions_cnt`：当天 unique sessionId 数
- `events_per_session_mean`：events_total / sessions_cnt
- `session_dur_mean_min` / `session_dur_p90_min`：按 session 的 (max(time)-min(time)) 统计
- `long_session_flag`：是否存在超长 session（例如 > 2h）

### 4.5 错误与质量

`status==404` 极少，但仍可保留：

- `http_404_cnt`、`http_404_ratio`
- `error_page_cnt`（page == 'Error'）与 HTTP 404 不同，建议同时保留

### 4.6 时间相关（“节奏”和“新老用户”）

- `days_since_registration`：当天距 registration 的天数（对新用户很关键）
- `days_since_prev_active`：距上一活跃日的间隔（只保留活跃日时非常重要）
- `dow`（day of week）：星期几（可以 one-hot 或 embedding）

---

## 5. 标签对齐与防泄露（建议明确两种训练目标）

你的现状标签（`prepare_datasets`）是：

- `label(user)=1` 当且仅当该用户在 **原始 train** 里出现过 `Cancellation Confirmation`

在日频化时，有两种训练口径：

### 口径 A：复刻当前 Kaggle 风格（最贴近线上 test 分布）

- 标签：仍然按 `Cancellation Confirmation` 是否出现定义
- 特征：删除 `Cancel`/`Cancellation Confirmation` 事件后再做日聚合
- 不额外做“提前 N 天预测”，允许模型利用“接近取消前”的行为（但看不到取消事件本身）

优点：与 test（缺少取消页面）更一致；上分概率更高。  
缺点：不一定符合“提前 10 天预警”的业务直觉。

### 口径 B：做“提前 N 天预警”的真实预测（更业务）

在 daily 层面更容易做到：

- 先算 `churn_date = min(date | page == 'Cancellation Confirmation')`
- 选择 `horizon_days = 10`
- 训练时只使用 `date <= churn_date - horizon_days` 的日特征（正样本），负样本则使用各自最后日期往前推同样 horizon 的截断

优点：严格无泄露、业务意义强。  
缺点：与 Kaggle test 分布可能不一致（需要看竞赛定义），得分可能下降但更真实。

建议：先跑口径 A（对齐现有 pipeline 的“竞赛得分”目标），再补口径 B 做对照实验。

---

## 6. 与现有代码的对接建议（最小改动路径）

建议在 `feature_pipeline.py` 新增一条并行管线（不要强改你现在事件级的实现）：

- 新增：`prepare_daily_datasets(...)`
  - 输入参数尽量与 `prepare_datasets` 对齐（含 cache、cutoff/horizon 等）
  - 输出：
    - `train_daily_df` / `val_daily_df` / `test_daily_df`：user-day 表（含 `userId`, `date`, daily 数值特征）
    - `labels`：user 级标签（同现状）
    - `artifacts`：列名、scaler、page 列集合等

Transformer 侧：

- 新增 `build_user_day_sequences(...)`（等价于 `build_user_sequences`，只是按 `date` 组织）
- 如果你决定“补齐日历”，可以让每个用户固定长度 51；否则照旧用 padding mask

---

## 7. 验收与对比实验（建议最少做 3 组）

1) 事件级（现有）作为 baseline  
2) 日频（只活跃日 + days_since_prev_active）  
3) 日频（补齐 51 天日历 + is_active_day）  

每组保持：

- 同一份 user 划分（train/val split）
- 同一套指标（AUC、F1、logloss）
- 同一条“显式泄露”检查：特征构造过程中确保 `Cancel`/`Cancellation Confirmation` 不进入聚合特征

