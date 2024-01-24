# vspliter

vspliter是为[GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)准备音频-文本对的工具

# Feature

- 利用whisper模型进行文本转录（openai/whisper-large-v3）（英文处理地很好，中文未测试）
- 通过自然语言分句来进行音频分割
- 每个句子有更好的开始和结束（不会没说完就掐断[绝大部分不会，土办法处理，不敢打包票])
- 用MDX23模型来分离人声和背景音

# How to use

本项目使用poetry环境，确保你安装了[poetry](https://python-poetry.org/docs/)

```
# 克隆项目
git clone https://github.com/GrahLnn/vspliter.git

# 进入项目目录
cd vspliter

# 激活并初始化 poetry 环境
poetry shell
poetry install

# 启动marimo notebook
marimo edit split.py
```

# Roadmap

- [ ] 日语的whisper识别有点问题，我的实验里它漏了将近40秒的内容（意味着有一个分割多了40秒的音频，但别的正常），要用的话记得检查一下
- [ ] 支持distil-whisper/large-v2（当前它的word-level时间戳不可用）
- [ ] 让音频的音量更加均匀
- [ ] 加入resemble-enhance进行降噪和增强
- [ ] 更多的配置项
