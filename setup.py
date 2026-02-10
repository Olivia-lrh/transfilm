from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="transfilm",
    version="0.1.0",
    author="Olivia-lrh",
    description="AI-powered video dubbing tool with Qwen3-ASR and Qwen3-TTS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Olivia-lrh/transfilm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "ffmpeg-python>=0.2.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.1",
        "modelscope>=1.11.0",
        "gradio>=4.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "pillow>=10.0.0",
    ],
    entry_points={
        "console_scripts": [
            "transfilm=cli:main",
            "transfilm-webui=webui:main",
        ],
    },
)
