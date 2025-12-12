# Transformer 模型特征与指标说明

## 一、问题与数据 & 目标概况

- 任务：利用用户的行为序列，预测该用户是否会流失（是否出现过页面 `Cancellation Confirmation`）。
- 备注：如需做“用更早的行为预测未来一段时间”的设定，可启用 `cutoff_time` + `label_mode=horizon`（严格不使用 cutoff 之后的数据做特征）。
- 样本定义：
  - 数据来自 `train.parquet` / `test.parquet`，每一行是一条用户行为事件。
  - 每个用户的行为序列长度不同（有的只有几条，有的达到一万多条）。
- 标签（churn）构造（基于原始 notebook & pipeline）：
  - 对 `train.parquet` 按 `userId` 聚合，只要该用户全历史序列中任意事件的 `page == "Cancellation Confirmation"`，则记为 `churn = 1`，否则为 `0`。
  - 对于最终发生取消的用户，`Cancellation Confirmation` 通常出现在该用户序列的最后一条或最后几条事件，这条信息不能直接用于特征（否则泄露标签）。
- 当前数据标签统计（train）：
  - 用户数：19,140
  - churn=1：4,271（约 22.3%）
  - churn=0：14,869（约 77.7%）
- 预测目标：
  - 输出每个用户发生 churn（出现 `Cancellation Confirmation`）的概率。

## 二、数据文件与范围（直接读取 parquet 统计）

### 2.1 文件规模

- `churn-prediction-25-26/train.parquet`
  - 行数：17,499,636
  - 用户数（`userId` 去重）：19,140
  - 时间范围（`time`）：2018-10-01 ~ 2018-11-20（共 51 个自然日）
- `churn-prediction-25-26/test.parquet`
  - 行数：4,393,179
  - 用户数（`userId` 去重）：2,904
  - 时间范围（`time`）：2018-10-01 ~ 2018-11-20（共 51 个自然日）

### 2.2 序列长度（每用户事件条数）

- train：min=1，p50=537，p90=2,245，p99=5,131，max=10,998
- test：min=1，p50=835，p90=3,085，p99=6,565，max=653,681（极端长序列用户：`userId=1261737`）

### 2.3 重要的 train/test 分布差异（做特征时要显式兼容）

- `auth`
  - train：`Logged In` / `Cancelled`
  - test：`Logged In` / `Logged Out` / `Guest`（train 没出现的类别）
- `page`
  - train 特有：`Cancel`、`Cancellation Confirmation`
  - test 特有：`Login`、`Register`、`Submit Registration`

## 三、字段信息记录（数据字典 + 取值形态）

> 目标：先把每列“有什么信息 / 是否稳定 / 是否可日频聚合 / 是否存在泄露风险”记录清楚。

### 3.1 字段清单（20 列）

> 注：parquet 里有一列 `__index_level_0__`，是 pandas 写入时的索引列，建模通常可直接丢弃。

| 列名 | 类型 | 信息/示例 | 观察与注意点 | 日频聚合建议 |
|---|---|---|---|---|
| `userId` | string | 用户 ID（提交文件的 `id`） | 主键；train/test 都存在 | 作为 group key |
| `time` | timestamp(us) | `2018-10-01 00:00:01` | 事件发生时间（无时区） | 取 `date=floor_day(time)` 作为日粒度 key |
| `ts` | int64 | `1538352001000` | 与 `time` 等价的 epoch(ms) | 一般用 `time` 即可 |
| `sessionId` | int64 | 会话 ID | 同一用户多 session；用于 session 特征 | 日内 distinct session 数、session 时长分布 |
| `itemInSession` | int64 | session 内序号 | 随 session 递增；可近似 session 事件长度 | session 内 max/mean，日内聚合 |
| `page` | string | `NextSong`/`Thumbs Up`… | 行为类型（核心离散列） | 日内按 page 计数、比例、是否出现 |
| `auth` | string | train: `Logged In`/`Cancelled`；test: `Logged Out`/`Guest` | **与标签强相关**（train 的 `Cancelled` 基本等同 churn 发生） | 日内计数（但要防泄露）；更建议直接过滤掉 `Cancelled` 事件 |
| `method` | string | `PUT`/`GET` | 请求方法，通常与 page 类型相关 | 日内 PUT/GET 计数、比例 |
| `status` | int64 | `200`/`307`/`404` | HTTP 状态码；`404` 与 `Error` page 高度相关 | 日内各状态计数、错误率 |
| `level` | string | `free`/`paid` | **会随时间变化**（train 约 52% 用户发生过切换） | 日内 level 众数/末次 level、paid 比例、切换指示 |
| `gender` | string | `M`/`F` | train/test 均稳定不变（按用户） | 静态特征（不必日频） |
| `registration` | timestamp(us) | `2018-08-08 13:22:21` | 按用户稳定不变 | 计算 tenure：`day - registration_day` |
| `location` | string | `New York-Newark-...` | 高基数（train 875 个）；按用户稳定不变 | 静态：解析州/地区；或做 target encoding/embedding |
| `userAgent` | string | 浏览器 UA | 低基数（train 85 个）；按用户稳定不变 | 静态：解析 OS/浏览器；或 embedding |
| `song` | string | 仅 `NextSong` 有值 | 非 `NextSong` 为 null（train 18.33% null；test 30.31% null） | 日内 unique song 数、重复率、top song 占比 |
| `artist` | string | 仅 `NextSong` 有值 | 同上 | 日内 unique artist、top artist 占比、集中度 |
| `length` | float | 歌曲时长（秒） | 仅 `NextSong` 有值；train 均值约 248.7s | 日内 sum/mean/std；总“听歌时长” proxy |
| `firstName` | string | 名字 | 高噪声/潜在泄露（身份信息） | 建议丢弃 |
| `lastName` | string | 姓氏 | 同上 | 建议丢弃 |
| `__index_level_0__` | int64 | parquet 索引列 | 无业务含义 | 丢弃 |

### 3.2 关键离散列的全量计数（train）

- `page`（共 19 类）
  - `NextSong` 14,291,433（81.67%）
  - `Thumbs Up` 789,391
  - `Home` 645,259
  - `Add to Playlist` 409,606
  - `Roll Advert` 284,837
  - `Add Friend` 262,147
  - `Logout` 204,700
  - `Thumbs Down` 164,964
  - `Downgrade` 124,248
  - `Settings` 101,191
  - `Help` 89,035
  - `Upgrade` 37,696
  - `About` 33,117
  - `Save Settings` 20,370
  - `Error` 17,294（与 `status=404` 数量一致）
  - `Submit Upgrade` 11,381
  - `Submit Downgrade` 4,425
  - `Cancel` 4,271
  - `Cancellation Confirmation` 4,271（= churn 用户数）
- `auth`
  - `Logged In` 17,495,365
  - `Cancelled` 4,271（极可能只在 churn 确认事件出现）
- `level`
  - `paid` 13,506,659
  - `free` 3,992,977
- `method`
  - `PUT` 16,162,688
  - `GET` 1,336,948
- `status`
  - `200` 16,020,693
  - `307` 1,461,649
  - `404` 17,294
- `length/song/artist` 缺失率：18.33%（对应非 `NextSong` 事件）

### 3.3 关键离散列的全量计数（test）

- `page`（共 20 类，补充了登录/注册相关行为）
  - `NextSong` 3,061,811
  - `Home` 481,816
  - `Login` 248,527
  - `Register` 649，`Submit Registration` 331
  - 其余与 train 基本一致（`Thumbs Up/Down`、`Add Friend`、`Roll Advert` 等）
- `auth`
  - `Logged In` 3,739,498
  - `Logged Out` 650,318
  - `Guest` 3,363
- `status`：`200` 3,829,565；`307` 559,169；`404` 4,445
- `length/song/artist` 缺失率：30.31%

### 3.4 按用户是否“稳定不变”（全量扫描结论）

- train（19,140 用户）
  - 始终不变：`gender`、`registration`、`location`、`userAgent`
  - 会变化：`level`（10,019 个用户发生过 `free/paid` 切换）

## 四、如何聚合成“日频（user-day）特征”

### 4.1 日频表的基本粒度与对齐方式

- 基本主键：`(userId, day)`，其中 `day = floor_day(time)`（自然日）
- 全局日历：2018-10-01 ~ 2018-11-20（51 天）
- 对齐策略（两种常用做法）
  1. **全窗口对齐**：对每个 `userId` 生成 51 天序列，缺失日填 0（便于 Transformer/TCN）
  2. **事件相对对齐**：以 `ref_day` 为锚点取最近 N 天
     - churn 用户：`ref_day = Cancellation Confirmation` 前一天（或前一时刻）
     - 非 churn：`ref_day = 2018-11-20`（或该用户最后活跃日）

### 4.2 先做“泄露过滤”（强烈建议写进 pipeline）

- 若 label 由 `page == "Cancellation Confirmation"` 构造，则这类事件（及其所在日）不能进入特征：
  - 过滤事件：`page in {"Cancellation Confirmation"}` 或 `auth == "Cancelled"`
  - 视你对“提前预警”的定义，可进一步过滤 `page == "Cancel"`（非常接近 churn 行为，容易造成近端泄露）

### 4.3 日内可直接 groupby 得到的基础特征（建议先把这些打全）

**A. 规模/活跃度**

- `d_events`：日内事件总数
- `d_sessions`：日内 distinct `sessionId`
- `d_pages_unique`：日内 distinct `page`
- `d_active_hours`：日内 distinct 小时数（`time.hour` 去重）
- `d_first_hour` / `d_last_hour`：日内首末事件小时

**B. 听歌（只针对 `page=="NextSong"`）**

- `d_nextsong_cnt`
- `d_listen_sec_sum`：`sum(length)`
- `d_listen_sec_mean` / `d_listen_sec_std`
- `d_song_unique`：distinct `song`
- `d_artist_unique`：distinct `artist`
- `d_repeat_song_rate = 1 - d_song_unique / d_nextsong_cnt`
- `d_artist_concentration`：top1 artist 播放占比（`max(artist_cnt) / d_nextsong_cnt`）

**C. 行为页计数（按 page 拆）**

建议为每个 `page` 做 `d_page_{name}_cnt`（未出现则 0），并再做一些比率类：

- 互动：`Thumbs Up`、`Thumbs Down`、`Add Friend`、`Add to Playlist`
  - `d_thumbs_up_rate = thumbs_up_cnt / d_nextsong_cnt`
  - `d_add_friend_per_session = add_friend_cnt / d_sessions`
- 广告：`Roll Advert`
  - `d_ad_rate = roll_advert_cnt / d_nextsong_cnt`
- 订阅意图：`Upgrade`、`Submit Upgrade`、`Downgrade`、`Submit Downgrade`
  - `d_upgrade_intent = 1[upgrade_cnt + submit_upgrade_cnt > 0]`
  - `d_downgrade_intent = 1[downgrade_cnt + submit_downgrade_cnt > 0]`
- 求助/设置：`Help`、`Settings`、`Save Settings`、`About`、`Home`、`Logout`

**D. 请求/错误**

- `d_status_404_cnt`、`d_status_307_cnt`、`d_error_page_cnt`
- `d_error_rate = (d_status_404_cnt + d_error_page_cnt) / d_events`
- `d_get_cnt`、`d_put_cnt`、`d_get_rate = d_get_cnt / d_events`

**E. 订阅层级（level）**

- `d_paid_ratio = count(level=='paid') / d_events`
- `d_level_last`：日内最后一条事件的 `level`（0/1）
- `d_level_changed_in_day`：日内是否出现过 free/paid 混合（可选）

### 4.4 先拆 session，再聚合回日内（能显著增强“行为结构”信息）

先构建 `(userId, sessionId)` 粒度的 session 表，然后按 day 聚合：

- `s_events`：session 内事件数
- `s_duration_sec`：`max(time) - min(time)`
- `s_nextsong_cnt`：session 内听歌数

再回到日内做统计量：

- `d_sess_events_mean/max/p90`
- `d_sess_duration_mean/max/p90`
- `d_sess_nextsong_mean/max`

### 4.5 跨日派生（在日频表上再做一层时序特征）

**A. Tenure/活跃间隔**

- `tenure_days = day - floor_day(registration)`
- `days_since_last_active`（按该用户上一次 `d_events>0` 的 day）
- `active_streak_days`：连续活跃天数

**B. Rolling/趋势（适合树模型或作为 Transformer 的额外输入）**

对关键计数/时长特征做滑窗（3/7/14 天）：

- `roll7_nextsong_sum/mean`
- `roll7_thumbs_up_sum`
- `roll7_ad_rate_mean`
- `trend7_nextsong = mean(last7) - mean(prev7)`

### 4.6 静态特征（建议与日频序列分开处理）

这些列按用户稳定不变，做成 user-level 静态向量更合适：

- `gender`（one-hot）
- `location`（解析州/城市/大区；或 embedding）
- `userAgent`（解析 OS/浏览器；或 embedding）
- `registration`（可转为 `registration_age_days_at_ref` 等）

### 4.7 归一化与稳健性建议

- 对计数/时长类用 `log1p` 或 clip（尤其要处理 test 的超长序列用户 `1261737`）
- 以 `d_nextsong_cnt` 做分母的比率特征要注意除零（无听歌日：设为 0 或缺失再填）
