#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行接口
"""

import os
import sys
import argparse
import logging
from transfilm.config import Config
from transfilm.pipeline import Pipeline
from transfilm.model_downloader import ModelDownloader
from transfilm.utils import setup_logging

logger = logging.getLogger(__name__)


def cmd_dub(args):
    """执行视频配音"""
    # 加载配置
    config = Config(args.config)
    
    # 设置日志
    log_level = args.log_level or config.get("logging.level", "INFO")
    setup_logging(level=log_level)
    
    logger.info(f"输入视频: {args.input}")
    logger.info(f"输出视频: {args.output}")
    logger.info(f"目标语言: {args.target_lang}")
    
    # 创建管道
    pipeline = Pipeline(config)
    
    # 进度回调
    def progress_callback(stage, total, message):
        print(f"[{stage}/{total}] {message}")
    
    try:
        # 运行管道
        output_path = pipeline.run(
            input_video=args.input,
            output_video=args.output,
            target_language=args.target_lang,
            source_language=args.source_lang,
            tts_speaker=args.speaker,
            progress_callback=progress_callback,
            cache_dir=args.cache_dir,
        )
        
        print(f"\n✅ 处理完成！")
        print(f"输出文件: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
        print(f"\n❌ 处理失败: {e}")
        return 1


def cmd_download(args):
    """下载模型"""
    # 加载配置
    config = Config(args.config)
    
    # 设置日志
    log_level = args.log_level or config.get("logging.level", "INFO")
    setup_logging(level=log_level)
    
    # 创建下载器
    downloader = ModelDownloader(
        source=args.source or config.get("download.source", "huggingface"),
        cache_dir=args.cache_dir or config.get("download.cache_dir", ""),
        show_progress=config.get("download.show_progress", True),
    )
    
    # 确定要下载的模型
    if args.models == ["all"]:
        models = list(ModelDownloader.SUPPORTED_MODELS.keys())
    else:
        models = args.models
    
    print(f"准备下载模型: {', '.join(models)}")
    print(f"下载源: {downloader.source}")
    
    try:
        # 下载模型
        model_paths = downloader.download_all_models(models)
        
        print("\n下载结果:")
        for model_key, path in model_paths.items():
            if path:
                print(f"  ✅ {model_key}: {path}")
            else:
                print(f"  ❌ {model_key}: 下载失败")
        
        return 0
        
    except Exception as e:
        logger.error(f"下载失败: {e}", exc_info=True)
        print(f"\n❌ 下载失败: {e}")
        return 1


def cmd_info(args):
    """显示信息"""
    # 加载配置
    config = Config(args.config)
    
    if args.speakers:
        # 显示可用说话人
        print("\n可用说话人:")
        speakers = config.get("speakers.custom_voice", [])
        for speaker in speakers:
            print(f"  - {speaker['name']}: {speaker['description']} ({speaker['language']})")
    
    elif args.languages:
        # 显示支持的语言
        print("\n支持的语言:")
        print("\nASR语言:")
        asr_langs = config.get("languages.asr_languages", [])
        for lang in asr_langs:
            print(f"  - {lang}")
        print("\nTTS语言:")
        tts_langs = config.get("languages.tts_languages", [])
        for lang in tts_langs:
            print(f"  - {lang}")
    
    elif args.models:
        # 显示支持的模型
        print("\n支持的模型:")
        for key, repo_id in ModelDownloader.SUPPORTED_MODELS.items():
            print(f"  - {key}: {repo_id}")
    
    else:
        # 显示通用信息
        print("\n=== Transfilm 视频配音系统 ===")
        print(f"版本: 1.0.0")
        print(f"\n配置文件: {args.config or '默认配置'}")
        print(f"\n使用 --help 查看更多命令")
    
    return 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Transfilm - AI视频配音系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 全局参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # dub 命令
    dub_parser = subparsers.add_parser("dub", help="视频配音")
    dub_parser.add_argument("input", type=str, help="输入视频路径")
    dub_parser.add_argument("--output", "-o", type=str, required=True, help="输出视频路径")
    dub_parser.add_argument("--target-lang", type=str, default="Chinese", help="目标语言")
    dub_parser.add_argument("--source-lang", type=str, default=None, help="源语言（自动检测）")
    dub_parser.add_argument("--speaker", type=str, help="TTS说话人")
    dub_parser.add_argument("--cache-dir", type=str, help="缓存目录")
    
    # download 命令
    download_parser = subparsers.add_parser("download", help="下载模型")
    download_parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=["all"] + list(ModelDownloader.SUPPORTED_MODELS.keys()),
        help="要下载的模型",
    )
    download_parser.add_argument("--source", type=str, choices=["huggingface", "modelscope"], help="下载源")
    download_parser.add_argument("--cache-dir", type=str, help="缓存目录")
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="显示信息")
    info_group = info_parser.add_mutually_exclusive_group()
    info_group.add_argument("--speakers", action="store_true", help="显示可用说话人")
    info_group.add_argument("--languages", action="store_true", help="显示支持的语言")
    info_group.add_argument("--models", action="store_true", help="显示支持的模型")
    
    # 解析参数
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # 执行命令
    if args.command == "dub":
        return cmd_dub(args)
    elif args.command == "download":
        return cmd_download(args)
    elif args.command == "info":
        return cmd_info(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
