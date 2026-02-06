# qig_agent
### 构建原因：

**一、对端到端训练的探索：“理解用户需求 ↔ 更好利用生图模型”的链路**
* 在模型训练和百科产线中，将 “用户Query → 工程化Prompt → 高质量生图” 形式化为带外部环境反馈的优化问题：通过prompt evaluation+image evalution双层评估，把“意图匹配度+画面质量+可用性”等下游效果回传为 reward，驱动policy model学习如何把query输入转写成更利于生图模型对齐且可控的工程化Prompt。对未来百度文生图模型训练方式进行探索。

**二、算法化闭环系统：GEPA 搜索 + PPO 蒸馏的双环优化，在线可控、离线可固化**
* 构建“运行时闭环 + 训练时双环”的可训练业务 Agent：
* 运行时用 PromptBank 检索提供高分先验与失败规避，策略模型采样生成 prompt 种子；
* 训练时用GEPA生成式进化搜索环最大化 reward，再用 PPO 将高奖励策略分布蒸馏固化到 Policy LLM，使策略从“依赖检索 + 多次试错”逐步过渡到“单次生成即可高准入”，实现工业场景的稳定性与成本可控（少抽卡，低试错）。

**三、数据资产与评测标准共建：**
* 为后续生图模型的优化训练提供高质量、合规化的标注训练数据，同时构建符合业务场景要求的标准化评测集，为模型性能验证与迭代提供可靠的数据支撑。
* 批量生成符合业务规范的高质量封面图、设计素材等，兼顾批量生产、品类多样性、输出质量稳定性、低试错成本（不抽卡）与高准入率，形成标准化生图解决方案，高效满足工业级生图诉求。

![系统架构图](assets/img/qig_agent技术报告_page-0001.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0002.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0003.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0004.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0005.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0006.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0007.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0008.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0009.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0010.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0011.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0012.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0013.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0014.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0015.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0016.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0017.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0018.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0019.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0020.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0021.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0022.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0023.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0024.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0025.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0026.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0027.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0028.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0029.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0030.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0031.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0032.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0033.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0034.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0035.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0036.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0037.jpg)
![系统架构图](assets/img/qig_agent技术报告_page-0038.jpg)
