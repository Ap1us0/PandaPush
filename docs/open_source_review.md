# PandaPush 开源源码调研

调研日期 2026-11-20。访问了 GitHub 仓库、固定提交的 Raw 源码以及 FetchPush 和 MuJoCo 建模文档。SB3 的 Read the Docs HER 页面经浏览工具访问失败，使用 GitHub 同版本实现与本地安装包交叉核对。

| 项目 | 实际阅读版本 | 许可证 |
| --- | --- | --- |
| [Gymnasium-Robotics](https://github.com/Farama-Foundation/Gymnasium-Robotics) | v1.4.2，`42eae53b2b27321d29090c219ce2f675c596de77` | MIT |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | v2.7.1，`8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458` | MIT |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | `370c43965323c0944b433b202aac5058b5f907a9`，与本地已有 Panda 资产一致 | Panda 子目录 Apache-2.0 |

## Gymnasium-Robotics

- `gymnasium_robotics/envs/fetch/push.py` 中 `MujocoFetchPushEnv.__init__` 选择 `fetch/push.xml`，关闭夹爪控制，每步 20 个仿真子步，成功半径 5 cm。场景包含桌面和自由关节方块。[FetchPush 源码](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/42eae53b2b27321d29090c219ce2f675c596de77/gymnasium_robotics/envs/fetch/push.py) · [任务文档](https://robotics.farama.org/envs/fetch/push/)
- `fetch_env.py` 的 `_set_action` 将前三维乘 0.05，第四维夹爪控制在 Push 禁用。Fetch 使用 mocap 增量，由 MuJoCo 内部的约束跟踪目标。PandaPush 输出三维增量，用显式六维 Jacobian IK 和七关节 PD 伺服驱动 Panda。[动作及观测源码](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/42eae53b2b27321d29090c219ce2f675c596de77/gymnasium_robotics/envs/fetch/fetch_env.py) · [mocap 工具](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/42eae53b2b27321d29090c219ce2f675c596de77/gymnasium_robotics/utils/mujoco_utils.py)
- `_get_obs` 返回字典，`achieved_goal` 为方块实际位置，`desired_goal` 为目标。 `_reset_sim` 恢复仿真状态并取与夹爪水平距离至少 0.1 m 的随机方块位置。 `_sample_goal` 在工作区取样，Push 将目标压到桌面方块中心高度。
- `compute_reward` 批量计算 L2 距离，稠密为负距离，稀疏越阈值为 -1。 `_is_success` 使用严格小于。 `BaseRobotEnv.step` 返回成功信息，普通任务通过 TimeLimit 截断。PandaPush 统一阈值边界，成功保持单独统计，用固定时限比较 HER。[环境接口](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/42eae53b2b27321d29090c219ce2f675c596de77/gymnasium_robotics/envs/robot_env.py)

## Stable-Baselines3

- `sac/sac.py` 的 `SAC.train` 更新双 Q、目标 Q、Actor 和自动熵系数。 `sac/policies.py` 的 `MultiInputPolicy` 直接消费 Dict 观测。本项目直接调用官方 `SAC.learn`、`predict`、`save` 和 `load`。[SAC 更新](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/sac/sac.py) · [策略](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/sac/policies.py)
- `DictReplayBuffer.add` 保存原始转移及 timeout，`HerReplayBuffer.sample` 混合真实经验和虚拟经验，`_sample_goals` 在同一已结束回合按 future 规则取达成目标。 `_get_virtual_samples` 同时替换当前和下一状态的目标，用下一状态的位置调用环境的批量 `compute_reward`。因此相对目标向量在特征提取时计算，重标时自动一致。 [普通缓冲](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/buffers.py) · [HER 缓冲](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/her/her_replay_buffer.py)
- `OffPolicyAlgorithm.save_replay_buffer/load_replay_buffer` 保存经验并在恢复时绑定新环境。官方 callback 提供定期保存和评估模式。项目使用自定义 callback 以复用固定的场景种子及方块接触指标，使用 SB3 logger 及 Monitor 落盘。多 Seed 由独立命令、目录、配置和模型管理，评估种子对齐。[续训接口](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/off_policy_algorithm.py) · [Callbacks](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/callbacks.py) · [评估接口](https://github.com/DLR-RM/stable-baselines3/blob/8da3e5eadeda14c63f42c62dbbe7dbb00c2fd458/stable_baselines3/common/evaluation.py)

## Menagerie 与本地复用

- 阅读 `panda.xml`、`panda_nohand.xml`、`README.md` 和 `LICENSE`。七个转动关节有不同角度、驱动力矩范围及伺服增益。视觉 OBJ 与碰撞几何分离，指尖碰撞垫和手臂碰撞网格参与接触。先前项目使用相同提交并记录了资产 Git 树 `386f680ac533760a49cb91933345949ad186b4c7`。[模型目录](https://github.com/google-deepmind/mujoco_menagerie/tree/370c43965323c0944b433b202aac5058b5f907a9/franka_emika_panda)
- 原样复制 `panda.xml`、`scene.xml`、`assets/`、`README.md`、`LICENSE`。保存 Apache-2.0 许可、上游来源及署名。推动器是运行时加在 hand 上的实体碰撞球，方块、目标区域、桌子与摩擦参数由独立的 MJCF 定义。MuJoCo 按几何优先级及显式 pair 解析摩擦。[MuJoCo 建模手册](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)
- 私有的同工作区工程 `机械臂强化学习/panda_reach` 提供 `panda_cartesian.py` 和 `panda_position.py`，其原始定义来源为 `轨迹规划/src`。复用位置伺服验证、Jacobian、姿态误差和力矩到控制量的转换，适配工具偏移与实时迭代预算。现有项目根目录未发现代码许可文件，按同一工作区用户授权内部复用，未赋予外部再分发许可。详细文件来源见 `code-source.json`。
- Fetch 与 SB3 的 MIT 实现作为设计参考或通过安装包调用。源码没有复制进入项目，发行这些依赖时应保留各自的许可和版权声明。

## 方案

自研 PandaPush 任务、接触监测、独立评估、随机化协议和报告。保留真实方块自由度、Panda 的可检查控制链路及 Fetch 式目标接口。稠密奖励帮助寻找从目标反方向接近方块的动作，稀疏对照完全依靠方块位置。成功依据实际方块到目标的三维误差，以及选定的保持步数。SAC、HER 和 PyTorch 使用官方实现。


## 接口和控制对比

| 实现 | 场景和观测 | 动作到物理运动 | 奖励和结束 |
| --- | --- | --- | --- |
| FetchPush v4 | Fetch、闭合平行夹爪、自由方块、目标字典 | 四维连续动作，XYZ 增量乘 0.05，推动时禁用夹爪分量。mocap–手体约束跟踪 | 5 cm 半径，负距离或 0/−1，注册时限 50 步 |
| Menagerie Panda | Panda MJCF、关节、视觉及碰撞网格 | 七路关节位置伺服与一路夹爪腱驱动，定义控制量和力矩范围 | 任务层决定奖励和结束 |
| SB3 SAC / HER | 使用外部环境的 Dict 观测 | Actor 输出连续动作，训练由 Replay Buffer 提供转移 | 奖励由环境提供，HER 用重标目标重新计算奖励 |
| 本项目 PandaPush | Panda、腕部推动球、自由方块、状态与目标字典 | 三维增量，最大 4.5 cm。六维 IK 生成关节参考，限幅 PD 伺服与接触求解 | 稠密或稀疏奖励，3 cm/3 步判定，60 步时限及安全终止 |

[FetchPush 场景](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/42eae53b2b27321d29090c219ce2f675c596de77/gymnasium_robotics/envs/assets/fetch/push.xml) · [Fetch 注册代码](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/42eae53b2b27321d29090c219ce2f675c596de77/gymnasium_robotics/__init__.py)

