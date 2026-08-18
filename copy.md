MOPD （Multi-Teacher On-Policy Distillation)的原理解析与实现-含代码
MOPD（Multi-Teacher On-Policy Distillation，多教师 On-Policy Distillation）可以理解成 多个 Teacher 模型共同指导一个 Student 模型，并且 Teacher 不是只对固定离线数据提供监督，而是针对 Student 当前 Policy 自己产生的 On-Policy Trajectory 进行蒸馏，它主要解决的是单 Teacher 蒸馏存在能力上限的问题，因为不同 Teacher 往往擅长不同类型的任务，例如一个 Teacher 擅长数学推理，一个 Teacher 擅长 Coding，一个 Teacher 擅长 Tool Use 和 Agent Planning，如果直接让 Student 学习某一个 Teacher，就容易把 Student 的能力限制在单一 Teacher 的分布上，而 MOPD 的核心思想就是让 Student 自己进行 Rollout，然后根据当前任务或者 trajectory 状态选择多个 Teacher 提供指导，再把这些 Teacher 的概率分布、Logits、Reasoning 行为或者最终轨迹转化成蒸馏信号，最终优化 Student，使 Student 在自己的 On-Policy 数据分布上逐渐吸收多个 Teacher 的能力，因此可以把 MOPD 理解成 Student Policy → On-Policy Rollout → Multi-Teacher Inference → Teacher Signal Aggregation → Distillation Loss → Student Update → New Student Policy，这里最关键的两个词就是 Multi-Teacher 和 On-Policy，Multi-Teacher 解决的是“多个专家能力如何融合”，On-Policy 解决的是“Teacher 如何针对 Student 当前真正容易犯错的状态提供监督”，因此它特别适合 AgentRL、Reasoning Model 和 Long-Horizon Agent 的后训练。
# MOPD最核心的训练结构
for task in tasks:

    # Student先自己执行任务
    trajectory = student.rollout(task)

    # 多个Teacher分别观察Student当前轨迹
    teacher_outputs = []

    for teacher in teachers:
        output = teacher.generate(
            task=task,
            trajectory=trajectory
        )
        teacher_outputs.append(output)

    # 聚合多个Teacher的监督信号
    target = aggregate_teacher_outputs(teacher_outputs)

    # Student根据Teacher信号进行蒸馏
    loss = distillation_loss(
        student,
        target
    )

    loss.backward()
    optimizer.step()
MOPD 和普通 Knowledge Distillation 最大的区别在于普通 KD 往往是 Teacher → Fixed Dataset → Student，Teacher 对数据集中的固定输入生成 Soft Label，然后 Student 学习 Teacher，而 MOPD 是 Student → Current State / Trajectory → Teacher → Student，也就是说 Teacher 看到的是 Student 当前真正遇到的问题，例如 Student 在 Agent Task 中已经正确调用了三个工具，但是在第四步 Observation 理解错误，那么 Teacher 不需要重新解决整个任务，而是可以直接针对这个状态提供下一步正确 Action，这种方式对于 Agent 特别有效，因为 Agent 的困难通常不是“整个任务完全不会”，而是长链路中的某一个局部状态出现错误，因此 On-Policy Distillation 可以把训练信号集中到 Student 当前的 Failure State。
# Student当前Trajectory
trajectory = [
    {"action": "search", "observation": "..."},
    {"action": "open_page", "observation": "..."},
    {"action": "extract", "observation": "..."},
    # Student在这里出现错误
]

# Teacher直接针对当前状态提供指导
teacher_action = teacher.generate_next_action(trajectory=trajectory)

# Student学习Teacher在当前State下的行为
loss = cross_entropy(
    student.next_action_logits,
    teacher_action
)
多 Teacher 的关键问题不是简单地把多个 Teacher 的输出平均，因为不同 Teacher 对同一个任务的可靠程度不同，例如 Teacher A 擅长 Coding，Teacher B 擅长数学，Teacher C 擅长 Agent Tool Use，如果当前任务是 Coding，那么 Teacher A 的信号应该拥有更高权重；如果当前状态是 Tool Calling，则应该提高 Tool-use Teacher 的权重，因此 MOPD 通常需要一个 Teacher Selection / Teacher Routing / Teacher Weighting 机制，根据 Task、State、Trajectory、Teacher Confidence 或历史表现动态决定哪些 Teacher 的信息应该进入最终 Distillation Target。
# 根据任务类型给不同Teacher分配权重
def teacher_router(task):

    if task.type == "coding":
        return {
            "coding_teacher": 0.7,
            "reasoning_teacher": 0.2,
            "agent_teacher": 0.1
        }

    if task.type == "math":
        return {
            "coding_teacher": 0.1,
            "reasoning_teacher": 0.8,
            "agent_teacher": 0.1
        }

    # Agent任务提高Agent Teacher权重
    return {
        "coding_teacher": 0.1,
        "reasoning_teacher": 0.2,
        "agent_teacher": 0.7
    }
如果 Teacher 都能够提供 Token-Level Logits，那么最直接的 MOPD 实现就是 Logit Distillation，也就是多个 Teacher 分别得到 next-token probability distribution，然后根据 Teacher Weight 对这些 Distribution 做加权融合，再让 Student 拟合最终 Distribution，例如 Teacher A 给出 P_A(y|x)，Teacher B 给出 P_B(y|x)，Teacher C 给出 P_C(y|x)，那么可以构造 P_M(y|x)=w_A P_A(y|x)+w_B P_B(y|x)+w_C P_C(y|x)，然后使用 KL Divergence 让 Student 的 Distribution 接近 Multi-Teacher Distribution。
import torch
import torch.nn.functional as F

# Student Logits
student_logits = student(input_ids).logits

# 多个Teacher Logits
teacher_logits = [
    teacher_a(input_ids).logits,
    teacher_b(input_ids).logits,
    teacher_c(input_ids).logits
]

# Teacher权重
weights = [0.5, 0.3, 0.2]

# 转换成概率分布
teacher_probs = [
    F.softmax(logits / temperature, dim=-1)
    for logits in teacher_logits
]

# 加权融合多个Teacher
target_probs = sum(
    w * p
    for w, p in zip(weights, teacher_probs)
)

# Student概率分布
student_log_probs = F.log_softmax(
    student_logits / temperature,
    dim=-1
)

# KL Distillation Loss
loss = F.kl_div(
    student_log_probs,
    target_probs,
    reduction="batchmean"
)
但在真正的大模型 MOPD 中，Teacher 往往比 Student 大很多，因此不能简单地在同一台 GPU 上同时加载多个完整 Teacher，否则显存和计算成本会非常高，所以工程上经常采用 Teacher Inference Service，Teacher 集群负责生成 Logits / Responses / Trajectories，Student Trainer 负责训练，二者通过数据流或者 RPC / 文件系统进行解耦，这样可以让多个 Teacher 并行 Rollout，同时 Student 使用已经缓存的 Teacher Signal 进行训练。
# Teacher服务端负责生成监督信号
class TeacherServer:

    def generate(self, request):

        outputs = []

        # 多Teacher并行推理
        for teacher in self.teachers:
            outputs.append(
                teacher.generate(request)
            )

        # 返回Teacher监督信息
        return aggregate(outputs)


# Student训练端只接收Teacher Target
target = teacher_server.generate(
    request
)

loss = student_distill_loss(
    student,
    target
)
对于 Agentic RL，MOPD 更重要的不是单纯蒸馏最终答案，而是蒸馏 Trajectory-level Policy，因为 Agent 的能力实际上包含 什么时候调用 Tool → 调用哪个 Tool → 参数怎么写 → 如何理解 Observation → 是否继续执行 → 什么时候停止 → 如何从错误中恢复，如果只蒸馏最终 Answer，Student 可能得到正确答案，却不会学习完成任务所需要的 Agent Policy，因此可以让多个 Teacher 对 Student 的完整 trajectory 进行指导，然后只对关键 Action 或关键 State 计算 Distillation Loss。
# Student产生完整Agent轨迹
student_traj = student_agent.rollout(task)

# Teacher针对每个关键State提供Action
targets = []

for state in student_traj.states:

    teacher_actions = [
        teacher.next_action(state)
        for teacher in teachers
    ]

    # 聚合多个Teacher的Action
    target_action = vote_or_weight(
        teacher_actions
    )

    targets.append(target_action)

# Student学习Teacher在关键State上的Policy
loss = policy_distillation_loss(
    student_traj,
    targets
)
这里可以进一步引入 Failure-aware MOPD，也就是不要让 Teacher 对 Student 的所有状态都提供同样强度的监督，而是重点蒸馏 Student 当前失败、低 Reward、Uncertain 或高风险的状态，例如 Student 在 100 个 Step 中只有 Step 37 出现错误，那么可以提高 Step 37 附近的 Distillation Weight，这样 Teacher Compute 可以集中在真正有价值的区域，也能够减少完整轨迹全部蒸馏带来的计算成本。
# 根据Student当前状态计算蒸馏权重
def get_distill_weight(state):

    weight = 1.0

    # Reward低的状态重点蒸馏
    if state.reward < 0:
        weight *= 3.0

    # Tool Error重点蒸馏
    if state.tool_error:
        weight *= 4.0

    # Student不确定时重点蒸馏
    if state.entropy > ENTROPY_THRESHOLD:
        weight *= 2.0

    return weight
MOPD 还可以结合 Teacher Confidence，因为不是所有 Teacher 在所有问题上都可靠，例如 Teacher A 对 Coding 的答案非常确定，而 Teacher B 对同一个 Coding 问题概率分布非常平坦，那么就应该降低 Teacher B 的权重，因此可以使用 Teacher Entropy、Log Probability、Verifier Score 或历史 Task Success Rate 估计 Teacher Reliability。
# 根据Teacher自身Confidence计算权重
def teacher_weight(output):

    # LogProb越高说明Teacher越确定
    confidence = output.mean_logprob

    # 可以进一步结合Verifier结果
    if output.verified:
        confidence *= 1.5

    return confidence
对于 AgentRL，一个非常实用的设计是让 Teacher 不负责最终 Reward，Teacher 负责提供 Policy Prior，而 Environment / Verifier 负责提供真实 Reward，因为 Teacher 本身可能犯错，如果完全相信 Teacher，就会把 Teacher 的错误蒸馏给 Student，而 Environment Success 才是真正的 Ground Truth，所以可以将 Loss 设计成 L = L_MOPD + λ L_RL，其中 L_MOPD 负责让 Student 学习多个 Teacher 的能力，L_RL 负责让 Student 最终服从 Environment Reward，这样 Teacher 更像一个高质量 Policy Prior，而不是绝对正确的 Oracle。
# Teacher蒸馏Loss
distill_loss = multi_teacher_kl(
    student_logits,
    teacher_logits,
    teacher_weights
)

# 环境Reward产生的RL Loss
rl_loss = grpo_loss(
    student,
    trajectories,
    rewards
)

# 联合训练
loss = (distill_loss + alpha * rl_loss)

loss.backward()
optimizer.step()
在 AgentRL 冷启动中，MOPD 甚至可以和 RFT 组合使用，先让 Student 进行 On-Policy Rollout，再让多个 Teacher 对这些轨迹进行评估和修正，然后使用 Verifier 过滤出最终成功的轨迹，成功轨迹进入 RFT，Teacher Policy Distribution 进入 MOPD，最后再进入 GRPO/PPO，这样就形成 Student Rollout → Multi-Teacher Guidance → Corrected Trajectory → RFT/MOPD → AgentRL，其中 RFT 主要学习高质量完整轨迹，MOPD 主要学习 Teacher 的细粒度 Policy Distribution，而 RL 最终负责让 Student 适应真实 Environment Reward。
# Student先进行On-Policy Rollout
trajectories = student.rollout(tasks)

for traj in trajectories:

    # 多Teacher分析Student当前轨迹
    teacher_signals = [
        teacher.analyze(traj)
        for teacher in teachers
    ]

    # Verifier判断最终任务结果
    reward = verifier.score(traj)

    if reward > 0:

        # 成功轨迹进入RFT
        rft_dataset.add(traj)

    # Teacher Policy进入MOPD
    mopd_dataset.add({
        "trajectory": traj,
        "teacher_signals": teacher_signals,
        "reward": reward
    })

# 先进行MOPD/RFT
student = train_mopd(student, mopd_dataset)
student = train_rft(student, rft_dataset)

# 最后进入AgentRL
student = train_grpo(student, environment)
真正工程化实现 MOPD 时，还需要考虑 Teacher-Student Vocabulary、Tokenizer、Sequence Alignment 和 Logit Storage，如果 Teacher 和 Student 使用不同 Tokenizer，那么 Token-Level Logit 不能直接做 KL，需要先进行 Vocabulary Mapping 或改成 Response-Level / Action-Level Distillation；如果 Teacher 和 Student 使用同一个 Tokenizer，则可以直接进行 Token-Level Distillation；如果多个 Teacher 的 tokenizer 不同，则 Agent 场景下更常见的工程方案是让 Teacher 输出标准化的 Action / Tool Call / Response，然后 Student 在自己的 Tokenizer 空间中学习这些结果，这种方式比强制多个不同模型做 Vocabulary-level KL 更容易落地。
# 如果Teacher和Student使用相同Tokenizer
teacher_logits = teacher(input_ids).logits
student_logits = student(input_ids).logits

loss = kl_divergence(
    student_logits,
    teacher_logits
)

# 如果Tokenizer不同
# 不直接进行Token-Level KL
# 而是蒸馏标准化后的Action/Response
teacher_action = teacher.generate_action(state)

loss = action_level_loss(
    student,
    state,
    teacher_action
)
从数据工程角度，MOPD 最重要的数据不是传统意义上的静态 prompt-answer，而是 task + student_state + student_action + observation + teacher_signal + reward + verifier_result，因为只有保存这些信息，后续才能重新分析 Teacher 为什么选择某个 Action、Student 为什么失败、哪个 Teacher 在这个 State 上更可靠，以及最终的 Reward 是否支持 Teacher 的判断，这也是 MOPD 和简单 Multi-Model Ensemble 最大的区别：Ensemble 通常是在推理阶段让多个模型一起回答然后投票，而 MOPD 是在训练阶段利用多个 Teacher 的知识构造 Student Policy Update，最终目标是把多个 Teacher 的能力压缩进一个 Student，而不是让线上每次请求都运行多个 Teacher。
# MOPD推荐的数据结构
sample = {
    "task": task,

    # Student当时看到的状态
    "state": state,

    # Student原始Action
    "student_action": student_action,

    # 环境Observation
    "observation": observation,

    # 多Teacher输出
    "teacher_signals": {
        "teacher_a": teacher_a_output,
        "teacher_b": teacher_b_output,
        "teacher_c": teacher_c_output
    },

    # Teacher权重
    "teacher_weights": {
        "teacher_a": 0.5,
        "teacher_b": 0.3,
        "teacher_c": 0.2
    },

    # Environment Reward
    "reward": reward,

    # 最终Verifier结果
    "verified": verified
}
最终可以把 MOPD 放在 AgentRL 后训练体系中的位置理解为：SFT 解决“模型从零学习基本 Agent 行为”，RFT 解决“从 Student 自己产生的轨迹中筛选成功行为并放大”，MOPD 解决“让多个更强 Teacher 在 Student 自己的 On-Policy State 上提供细粒度 Policy Guidance”，AgentRL 则解决“最终让 Policy 直接根据 Environment Reward 优化长期任务目标”，所以 MOPD 最核心的创新并不是简单的“多个 Teacher 一起蒸馏”，而是 Multi-Teacher + On-Policy + State/Trajectory-level Guidance + Dynamic Teacher Weighting + Verifier/Reward Alignment，它让 Teacher 不再只对静态 Dataset 讲课，而是针对 Student 当前真正遇到的困难提供指导，再通过 Distillation Loss 将多个 Teacher 的能力压缩到 Student 中；如果进一步和 RFT、GRPO 结合，就可以形成 SFT → On-Policy Rollout → Multi-Teacher Guidance → MOPD/RFT → Positive Trajectory Density提升 → GRPO/PPO AgentRL → Evaluation → Bad Case Mining → 新一轮On-Policy MOPD 的闭环，这也是 MOPD 在大模型 Agent 后训练中真正有价值的地方。