# vspliter

vspliter是为GPT-SoVITS训练准备音频数据的工具

# Feature

- 利用whisper模型进行文本转录（openai/whisper-large-v3）（英文处理地很好，中文未测试）
- 通过自然语言分句来进行音频分割
- 每个句子有更好的开始和结束（不会没说完就掐断[大部分应该不会])
- 用MDX23模型来分离人声和背景音

# How to use

clone本项目

本项目使用poetry环境，确保你安装了[poetry](https://python-poetry.org/docs/)

```
cd vspliter
poetry shell
poetry install
marimo edit split.py
```

# Roadmap

- [ ] 日语的whisper识别有点问题，我的实验里它漏了将近40秒的内容
- [ ] 支持distil-whisper/large-v2，（当前word-level的时间戳不可用）
- [ ] 让音频的音量更加均匀
- [ ] 加入resemble-enhance进行降噪和增强
- [ ] 更多的配置项（用来判断是否能对模型进行加速的操作）