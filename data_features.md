# Transformer 模型特征与指标说明

## 一、问题与数据 & 目标概况

- 任务：在给定观测窗口内（约截止到 2018-11-20），利用用户的行为序列，预测该用户在之后 10 天内是否会流失（访问页面 `Cancellation Confirmation`）。
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
  - 输出每个用户在未来 10 天窗口内发生 “Cancellation Confirmation” 的概率，即 churn 概率。

---

## 二、Transformer 端当前的特征工程概览

这一部分主要来自 `feature_pipeline.py` 和 `transformer_model.py`，只关注 Transformer 实际使用到的特征。

### 2.1 标签与时间截断策略

- 标签：
  - 使用原始 `train.parquet`，按 `userId` 聚合，只要全序列中出现过 `Cancellation Confirmation`（不管发生在何时），就标记为 churn 用户。
- 防止信息泄露的处理：
  - 在构建特征时，从 `train_raw` 中删去 `page == "Cancellation Confirmation"` 和 `page == "Cancel"` 的所有记录。
  - 使用 `truncate_user_histories`：
    - 对每个用户按 `time` 排序后，删除序列最后一小段（按 `buffer_min=2`, `buffer_frac=0.1`），避免模型看到“取消动作前一两步”的明显信号。
  - 可选 `cutoff_time` 和 `truncate_last_days_per_user`：
    - 可以按绝对时间或者“每个用户最后 N 天”进一步裁剪行为，用于模拟“用更早的行为预测未来一段时间”的场景（当前默认未强制启用）。

### 2.2 事件级基础清洗与编码

- 删除的字段：`firstName`, `lastName`, `ts`, `auth`, `itemInSession`, `sessionId`, `method`（对 churn 预测贡献较小或难以泛化）。
- HTTP 状态码 `status`：
  - 构造特征 `error occur`：
    - 对每个用户、按 `time` 排序，计算“当前之前出现过多少次 404 / 当前事件序号”。
    - 得到一个随时间变化的“历史 404 比例”序列，反映用户使用过程中的错误体验质量。
  - 使用该比例作为数值特征，原始的 `status` 字段随后被删除。
- `gender`：
  - 使用映射 `F -> 0`, `M -> 1`，缺失值填 0，作为整数型特征。
- `level`：
  - 使用映射 `free -> 0`, `paid -> 1`，缺失值填 0，用于区分是否付费用户。

### 2.3 页面（page）相关处理（当前 Transformer 已使用部分）

- 原始 `page` 取值示例及频次：
  - `NextSong`（最多）、`Thumbs Up`, `Home`, `Add to Playlist`, `Roll Advert`, `Add Friend`, `Logout`, `Thumbs Down`, `Downgrade`, `Settings`, `Help`, `Upgrade`, `About`, `Save Settings`, `Error`, `Submit Upgrade`, `Submit Downgrade`, `Cancellation Confirmation`, `Cancel` 等。
- 在特征工程中：
  - 先对 `page` 做 one-hot（虚拟变量），但在 `prepare_datasets` 中会整体删除这批 one-hot 列，以避免维度过大。
  - 构建整数索引 `page_id`：
    - 基于 train+test 里的 `page` 取值组合成 `page_categories`。
    - 对每种 page 分配一个从 1 开始的 ID（0 预留给 padding/未知）。
- 在 Transformer 模型中：
  - 使用 `page_id` 通过 `nn.Embedding(num_pages + 1, 32)` 转为 32 维 embedding。
  - 这样，页面种类信息以稠密向量形式参与序列建模，而不是用高维 one-hot。

### 2.4 地理与设备特征

- `location` 被拆成两个字段：
  - `metro`：城市/都会区，如 “Dallas-Fort Worth-Arlington”。
  - `state`：州，如 “TX”。
- `userAgent` → `device`：
  - 利用简单字符串匹配解析设备类别：`iPhone`, `iPad`, `Mac`, `Windows`, `Android`, `Linux`, 其他统一为 `Other`。
- 将上述三个字段映射为整数 ID：
  - `metro_id`, `state_id`, `device_id`：
    - 根据 train+test 中的取值构建映射字典，未知值为 0。
- 在 Transformer 中：
  - `metro_id` → 16 维 embedding。
  - `state_id` → 12 维 embedding。
  - `device_id` → 6 维 embedding。

### 2.5 歌曲与歌手的消费强度与集中度

通过 `get_song_stats_fast` 函数，为每个用户按时间滚动计算以下特征（均为“到当前事件为止”的统计量）：

- 多样性相关：
  - `num_distinct_song_until_now`：当前为止听过的不同歌曲数量。
  - `num_distinct_artist_until_now`：当前为止听过的不同歌手数量。
- 单一歌曲偏好度：
  - `song_top1_count_until_now`：当前为止播放次数最多那首歌的播放次数。
  - `song_top1_frac_until_now`：这首“top1 歌曲”播放次数占所有播放次数的比例。
- 前三首歌曲集中度：
  - `song_top3_count_until_now`：前三首播放最多的歌曲之和的播放次数。
  - `song_top3_frac_until_now`：前三首播放最多的歌曲播放次数占总播放次数的比例。
- 歌手层面的类似统计：
  - `artist_top1_count_until_now` / `artist_top1_frac_until_now`：最常听歌手的次数及占比。
  - `artist_top3_count_until_now` / `artist_top3_frac_until_now`：前三位歌手播放次数之和及占比。

这些特征可以刻画用户的内容消费模式：

- 听歌是否非常集中在少数几首歌或少数歌手上；
- 是“广泛探索型”还是“重度依赖固定歌单型”；
- 这些模式随时间变化的趋势会被 Transformer 在序列中捕捉到。

### 2.6 时间相关特征

- `regis_time = time - registration`：
  - 表示当前事件距离用户注册时间的时间差（一个 timedelta 类型）。
- `regis_time_seconds`：
  - 在 `prepare_datasets` 中，将上面的时间差转换为秒数，并作为最终的数值特征之一。
- 原始 `time` 字段：
  - 保留用于排序和截断行为序列，但不直接作为数值特征输入模型（避免日期数值过大、难以解释）。

### 2.7 Transformer 实际使用的数值特征列表（事件级）

在删掉文本字段、one-hot 列等之后，通过 `_infer_numeric_cols` 推断数值列，并额外加入 `regis_time_seconds`，Transformer 端最终使用的大致事件级数值特征包括：

- 用户属性与简单行为：
  - `gender`：0/1。
  - `level`：0/1。
  - `error occur`：历史 404 比例（到当前事件为止）。
  - `length`：当前歌曲播放时长。
- 内容消费统计与集中度：
  - `num_distinct_song_until_now`
  - `num_distinct_artist_until_now`
  - `song_top1_frac_until_now`
  - `song_top1_count_until_now`
  - `song_top3_frac_until_now`
  - `song_top3_count_until_now`
  - `artist_top1_frac_until_now`
  - `artist_top1_count_until_now`
  - `artist_top3_frac_until_now`
  - `artist_top3_count_until_now`
- 时间相关：
  - `regis_time_seconds`：当前事件距注册的时间长度（秒）。
- 所有数值特征会在用户级样本构建前经过 `StandardScaler` 标准化。

### 2.8 序列构造与 Transformer 输入形式

- 序列构造（`build_user_sequences`）：
  - 对每个用户按 `time` 排序后，保留最近 `max_seq_len = 400` 条事件。
  - 对每条事件提取：
    - 数值特征向量 `numeric[numeric_cols]`。
    - 类别 ID：`page_id`, `metro_id`, `state_id`, `device_id`。
- Batch 组装（`collate_batch`）：
  - 按 batch 对序列做 padding，生成：
    - `numeric_feats`: `[batch_size, seq_len, num_numeric]`。
    - `page_ids`, `metro_ids`, `state_ids`, `device_ids`: `[batch_size, seq_len]`。
    - `padding_mask`: `[batch_size, seq_len]`，用于告知 Transformer 哪些位置是 padding。
- 模型结构（`ChurnTransformer`）：
  - 将四个类别 embedding（page/metro/state/device）拼接后与数值特征拼接，通过线性层投到 `d_model = 128`。
  - 加入位置编码，输入多层 Transformer Encoder。
  - 利用 padding mask 做序列 mean pooling（只对真实事件位置求平均），得到用户级向量。
  - 最后通过 MLP 输出一个标量 logit，Sigmoid 后即为 churn 概率。

---

## 三、当前特征覆盖的信息维度总结

综合来看，当前 Transformer 端特征主要覆盖以下几个维度：

- 用户基本属性：性别（`gender`）、是否付费（`level`）、注册时间长短（`regis_time_seconds`）。
- 使用环境：城市/都会区（`metro_id`）、州（`state_id`）、设备类型（`device_id`）。
- 使用质量：HTTP 404 错误比例（`error occur`），反映客户端/网络/服务端问题是否频繁出现。
- 内容消费行为：
  - 听歌/听歌手的多样性（distinct 数量）。
  - 对少数歌曲/歌手的依赖程度（top1/top3 集中度与次数）。
- 页面行为：
  - 通过 `page_id` 的 embedding 表示当前行为的类型（听歌、加好友、点赞、设置、升级/降级、帮助、注销、取消等）。
  - 但目前还没有显式的“页面频率统计”或“页面路径模式”特征。
- 时间维度：
  - 注册时间维度（距离注册多久）。
  - 行为序列顺序本身（通过位置编码和历史滚动统计间接体现）。

---

## 四、基于当前 pages 与序列结构的下一步因子规划

下面是结合目前的数据情况（尤其是 `page` 列），在 Transformer 框架下适合进一步构建的一些候选因子，方便后续迭代。

### 4.1 Page 访问频率与流失路径相关因子

- 关键 page 的累积 & 近期访问次数 / 占比（事件级滚动窗口）：
  - 针对 `Help`, `Settings`, `Upgrade`, `Downgrade`, `Submit Upgrade`, `Submit Downgrade`, `Logout`, `Error`, `Add Friend` 等页面，构造：
    - 历史累计访问次数（`*_count_until_now`）。
    - 最近 K 次事件中的访问占比（`*_ratio_last_k_events`）。
  - 这可以帮助区分：
    - 正常“听歌型”行为 vs. “频繁设置/求助/修改套餐型”行为。
- 页面转移模式：
  - 事件级增加 `prev_page_id`（前一个事件的 page ID），并使用 embedding。
  - 这样 Transformer 可以学习类似 `Settings -> Help -> Cancel` 这样的路径结构，而不只是单点的 page 类型。
  - 还可以在特征工程阶段预先标出一些典型路径的标志位，例如：
    - 在某个窗口内是否出现过 `Help` 或 `Settings` 后紧接着 `Cancel` / `Cancellation Confirmation` 的序列（避免直接看 cancel 行本身，可用更早的行为）。

### 4.2 时间节奏与活跃度因子

- 活跃节奏：
  - `days_since_last_event` / `seconds_since_last_event`：事件间时间间隔，用于刻画用户最近是否“突然冷却”。
  - 滚动统计：过去 1、3、7 天内的事件数量或平均每天事件数（可以按用户+时间在特征工程中预先计算）。
- 时间标签 embedding：
  - 将 `time` 映射为：
    - `hour_of_day`（0~23），`day_of_week`（0~6）两类离散特征，加 embedding。
  - 让模型学习：
    - 是否只在某些时间段活跃、是否周末才听歌等模式。

### 4.3 等级（level）变化与付费路径

- 相比当前的“当前 level”，可以增加：
  - `level_change_flag`：与上一事件相比，level 是否发生变化。
  - `num_upgrade_events_until_now` / `num_downgrade_events_until_now`。
  - `events_since_last_level_change` 或 `days_since_last_level_change`。
- 对付费用户特别关注：
  - 近期是否频繁出现降级/取消相关页面（`Downgrade`, `Submit Downgrade`, `Help`, `Settings`）。
  - 例如构造“上一笔升级/降级到现在的时间/事件数”等特征，反映“刚升级就取消”这种不满意情况。

### 4.4 错误与广告负面体验相关因子

- 在已有 `error occur` 的基础上，增加：
  - `error_ratio_last_k_events`：最近一个窗口内的错误率，区分“历史上偶尔出错”和“最近集中出错”。
  - `error_burst_flag`：短时间内连续多次 `Error` / 404 的爆发事件。
- 广告与推送：
  - `roll_advert_count_until_now` / `roll_advert_ratio_last_k_events`：广告曝光的频率与占比。
  - 考虑与用户满意度行为（`Thumbs Up` / `Thumbs Down`）结合：
    - 构造“Thumbs Up/Down 比例”和其近期变化，观察是否在广告增多时负反馈升高。

### 4.5 内容消费与“跳歌”行为因子

- 结合 `NextSong` + `length` + 时间：
  - 利用连续两条 `NextSong` 事件的时间差与上一首歌 `length` 的对比，如果实际播放时间远小于歌曲长度，可以视为“跳歌”。
  - 构造：历史跳歌率、近期跳歌率、跳歌的 burst 特征（短时间内多次跳歌）。
- 在当前多样性/集中度特征基础上增加：
  - 平均歌曲时长（历史与近期），区分“短平快型”听歌 vs “完整听完型”。
  - 歌曲/歌手多样性的近期变化趋势（例如最近 K 次事件的 distinct/song ratio）。

### 4.6 适配 Transformer 的实现建议

- 尽量将新增因子设计为：
  - 事件级数值特征（继续进入 `numeric_cols`）；
  - 或少数类别特征（例如时间段、阶段等）+ embedding。
- 形态上以“滚动窗口统计 + 历史累计统计”为主：
  - 有利于 Transformer 在序列中学习趋势（例如 Help 占比是否突然升高）。
  - 保持与现有 pipeline 一致的风格，便于集成与调参。

---

## 五、小结

- 当前 Transformer 已经利用：
  - 性别、付费等级、错误率、播放时长、歌曲/歌手多样性与集中度、设备与地理位置、注册时间以及页面类别（通过 `page_id` embedding）。
- 尚未显式建模的重点方向：
  - 基于 `page` 的访问频率、路径模式与近期变化（Help/Settings/Upgrade/Downgrade/Logout/Error 等）。
  - 更细粒度的时间节奏（活跃间隔、日/周周期）与生命周期阶段。
  - 等级变化路径、错误爆发、广告负载以及“跳歌”行为等。
- 下一步可以优先围绕 `page + time` 做滚动统计与路径特征，再逐步加入 level 变化、错误/广告和跳歌因子，以充分发挥 Transformer 对序列模式建模的优势，提升 churn 预测能力。

