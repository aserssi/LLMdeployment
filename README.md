# 常见大语言模型分享比较

## 1 项目简介

登录并使用魔搭平台，关联阿里云账号来获得免费的CPU云计算资源；

通过Jupyter Notebook进入相应的项目部署环境，完成模型的部署；

针对2-3个不同的模型进行一些应用场景的测试，并开展不同模型之间的横向对比；

### 本项目中横向对比的几个模型分别是：

通义千问Qwen-7B-Chat

智谱ChatGLM3-6B

百川2-7B-对话模型

### 应用场景样例为：

请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少

请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上

他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道

明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？

领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。

## 2 配置流程
### 2.1 配置云服务器
### 2.2 环境配置
在终端命令行环境中输入下列命令下载并安装 Miniconda:
```
   cd /opt/conda/envs
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
   echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   conda --version
   ```
创建并激活专属环境:
```
   conda create -n qwen_env python=3.10 -y
   source /opt/conda/etc/profile.d/conda.sh
   conda activate qwen_env
   ```
安装必需的基础依赖库：
```
   pip install -U pip setuptools wheel
   pip install torch==2.3.0+cpu torchvision==0.18.0+cpu --index-url https://download.pytorch.org/whl/cpu
   pip install fschat --use-pep517
   pip install tqdm huggingface-hub
   ```
### 2.3 大语言模型下载
切换至工作目录：
```
   cd /mnt/workspace
   ```
下载对应大语言模型
```
   git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git 
   git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git 
   git clone https://www.modelscope.cn/baichuan-inc/Baichuan2-7B-Chat.git
   ```
### 2.4 构建与使用chatbot
在 Notebook 的 Launcher 中选择基于 itrex 的环境，新建一个.ipynb 文件
量化并构建 chatbot
```python
   # Build chatbot with INT4 weight-only quantization, computations in AMX INT8
   from intel_extension_for_transformers.neural_chat import build_chatbot, PipelineConfig
   from intel_extension_for_transformers.transformers import RtnConfig
   config = PipelineConfig(model_name_or_path="./chatglm3-6b",
   						optimization_config=RtnConfig(bits=4, compute_dtype="int8", 
   						weight_dtype="int4_fullrange"))
   chatbot = build_chatbot(config)
   ```
新建单元格，生成回答
```
   # Perform inference/generate a response
   response = chatbot.predict(query="请说出以下两句话区别在哪里?1.冬天：能穿多少穿多少；2.夏天：能穿多少穿多少；")
   print(response)
   ```
点击运行即可查看回答。可以将`response`中的`query`参数替换为其他语境下的问题。
