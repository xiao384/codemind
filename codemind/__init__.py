"""CodeMind: 从零实现的代码大模型与编程 CLI。

本包不依赖任何现成的预训练模型、推理框架或第三方 tokenizer：
分词器、Transformer、训练循环、采样器、CLI 全部为原创实现，仅依赖 PyTorch 与 NumPy。

模块地图::

    codemind.config     超参配置（模型 / 训练 / 数据）
    codemind.tokenizer  字节级 BPE 分词器（含 FIM 特殊词元）
    codemind.model      从零实现的 decoder-only Transformer
    codemind.data       代码语料采集、二进制分片、采样与 FIM 变换
    codemind.train      训练循环（AMP / 梯度累积 / 断点续训）
    codemind.generate   采样器与增量 KV-Cache 流式解码
    codemind.editing    文件级代码编辑工具（应用补丁、语法校验）
    codemind.cli        命令行入口
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
