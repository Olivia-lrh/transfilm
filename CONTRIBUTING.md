# Contributing to TransFilm / 为 TransFilm 贡献

感谢您对 TransFilm 项目的关注！我们欢迎各种形式的贡献。

Thank you for your interest in contributing to TransFilm! We welcome contributions of all kinds.

## 开发环境设置 / Development Setup

### 1. Fork and Clone / Fork 并克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/transfilm.git
cd transfilm
```

### 2. 安装依赖 / Install Dependencies

```bash
# 创建虚拟环境 / Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 安装依赖 / Install dependencies
pip install -r requirements.txt

# 安装开发依赖 / Install development dependencies
pip install -e .
```

### 3. 运行测试 / Run Tests

```bash
# 结构测试 / Structure tests
python test_structure.py

# 完整测试 / Full tests (requires all dependencies)
python test.py
```

## 代码规范 / Code Standards

### Python 代码风格 / Python Code Style

我们使用 Black 进行代码格式化，使用 Flake8 进行代码检查。

We use Black for code formatting and Flake8 for linting.

```bash
# 格式化代码 / Format code
black transfilm/ cli.py webui.py

# 检查代码 / Lint code
flake8 transfilm/ cli.py webui.py --max-line-length=100
```

### 提交信息 / Commit Messages

使用清晰的提交信息，遵循以下格式：

Use clear commit messages following this format:

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
style: Format code
refactor: Refactor code
test: Add tests
chore: Update dependencies
```

## 贡献类型 / Types of Contributions

### 🐛 报告错误 / Bug Reports

使用 GitHub Issues 报告错误，请包含：
- 错误描述 / Error description
- 复现步骤 / Steps to reproduce
- 期望行为 / Expected behavior
- 实际行为 / Actual behavior
- 环境信息 / Environment info (OS, Python version, GPU/CPU)

### ✨ 功能请求 / Feature Requests

我们欢迎新功能建议！请说明：
- 功能描述 / Feature description
- 使用场景 / Use cases
- 预期收益 / Expected benefits

### 📝 文档改进 / Documentation Improvements

文档改进始终受欢迎，包括：
- README 更新 / README updates
- 代码注释 / Code comments
- 使用示例 / Usage examples
- 翻译 / Translations

### 🔧 代码贡献 / Code Contributions

1. 创建新分支 / Create a new branch
```bash
git checkout -b feature/your-feature-name
```

2. 进行更改 / Make your changes
   - 编写清晰的代码 / Write clear code
   - 添加注释 / Add comments
   - 更新文档 / Update docs
   - 添加测试 / Add tests

3. 提交更改 / Commit changes
```bash
git add .
git commit -m "feat: Add your feature"
```

4. 推送到 GitHub / Push to GitHub
```bash
git push origin feature/your-feature-name
```

5. 创建 Pull Request / Create a Pull Request
   - 描述更改 / Describe changes
   - 链接相关 Issue / Link related issues
   - 等待审查 / Wait for review

## 开发指南 / Development Guidelines

### 项目结构 / Project Structure

```
transfilm/
├── transfilm/          # 核心包 / Core package
│   ├── __init__.py
│   ├── pipeline.py     # 主流程 / Main pipeline
│   ├── video_processor.py
│   ├── audio_processor.py
│   ├── asr_engine.py
│   ├── tts_engine.py
│   └── utils.py
├── cli.py             # 命令行界面 / CLI
├── webui.py           # Web界面 / Web UI
├── config.py          # 配置 / Configuration
└── tests/             # 测试 / Tests
```

### 添加新功能 / Adding New Features

1. 在适当的模块中实现功能 / Implement in appropriate module
2. 更新配置（如需要）/ Update config if needed
3. 添加到 CLI/WebUI（如需要）/ Add to CLI/WebUI if needed
4. 编写测试 / Write tests
5. 更新文档 / Update documentation

### 性能优化 / Performance Optimization

我们重视低显存环境下的性能：
- 使用模型卸载 / Use model offloading
- 支持量化 / Support quantization
- 优化内存使用 / Optimize memory usage
- 提供 CPU 回退 / Provide CPU fallback

## 测试 / Testing

### 单元测试 / Unit Tests

```bash
pytest tests/
```

### 集成测试 / Integration Tests

```bash
# 使用示例视频测试 / Test with sample video
python cli.py --input examples/sample.mp4 --output test_output.mp4
```

## 获取帮助 / Getting Help

- 📖 查看文档 / Check documentation: [README.md](README.md)
- 💬 提问 / Ask questions: GitHub Discussions
- 🐛 报告问题 / Report issues: GitHub Issues
- 📧 联系维护者 / Contact maintainer: [Create an issue]

## 许可证 / License

贡献的代码将使用 MIT 许可证。

Code contributions are licensed under the MIT License.

## 致谢 / Acknowledgments

感谢所有贡献者的努力！

Thanks to all contributors for their efforts!

---

再次感谢您的贡献！🎉

Thank you again for your contribution! 🎉
