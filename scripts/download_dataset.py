"""下载  原始数据，并生成不可伪造的文件清单。"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import requests

try:
    from .data_common import (
        config_project_root,
        file_specs,
        load_yaml,
        resolve_project_path,
        sha256_file,
        utc_now,
        write_yaml_atomic,
    )
except ImportError:  # 允许直接执行 python scripts/download_dataset.py
    from data_common import (  # type: ignore
        config_project_root,
        file_specs,
        load_yaml,
        resolve_project_path,
        sha256_file,
        utc_now,
        write_yaml_atomic,
    )


USER_AGENT = "amazon-esci--data-preparation/1.0"
REQUEST_TIMEOUT = (15, 120)


class DownloadError(RuntimeError):
    """ 下载或文件一致性错误。"""


def _load_existing_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = load_yaml(path)
    except Exception as exc:
        raise DownloadError("已有 MANIFEST.yaml 无法读取: %s" % exc) from exc
    return value


def _manifest_hash(manifest: Mapping[str, Any], name: str) -> Optional[str]:
    for item in manifest.get("raw_files", []) or []:
        if isinstance(item, dict) and item.get("name") == name:
            value = item.get("sha256")
            return value if isinstance(value, str) and value else None
    return None


def _download_to_file(url: str, destination: Path) -> None:
    """流式下载到临时文件，网络中断时用 HTTP Range 断点续传。"""

    temp_name = str(destination.with_name(".%s.part" % destination.name))
    completed = False
    try:
        for attempt in range(1, 6):
            offset = os.path.getsize(temp_name) if os.path.exists(temp_name) else 0
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = "bytes=%d-" % offset
            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                ) as response:
                    response.raise_for_status()
                    # 某些服务会忽略 Range；此时从头开始，避免把完整文件接在残片后面。
                    append = offset > 0 and response.status_code == 206
                    if offset > 0 and not append:
                        offset = 0
                    mode = "ab" if append else "wb"
                    downloaded = offset
                    with open(temp_name, mode) as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            output.write(chunk)
                            downloaded += len(chunk)
                            if downloaded and downloaded % (100 * 1024 * 1024) < len(chunk):
                                print("  已下载 %.1f MB" % (downloaded / 1024 / 1024), flush=True)
                        output.flush()
                        os.fsync(output.fileno())

                if downloaded == 0:
                    raise DownloadError("下载结果为空: %s" % url)
                os.replace(temp_name, str(destination))
                completed = True
                return
            except requests.RequestException as exc:
                print("  第 %d 次传输中断，将尝试续传: %s" % (attempt, exc), flush=True)
                if attempt == 5:
                    raise DownloadError("网络下载失败: %s (%s)" % (url, exc)) from exc
            except OSError as exc:
                raise DownloadError("写入下载文件失败: %s (%s)" % (destination, exc)) from exc
    finally:
        if not completed:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def _file_metadata(
    path: Path,
    name: str,
    url: str,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    if not path.is_file():
        raise DownloadError("原始文件不存在或不是普通文件: %s" % path)
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise DownloadError("原始文件为空: %s" % path)
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256:
        raise DownloadError(
            "SHA-256 不一致: %s，期望 %s，实际 %s"
            % (name, expected_sha256, digest)
        )
    return {
        "name": name,
        "url": url,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def build_manifest(
    config: Mapping[str, Any],
    files: List[Dict[str, Any]],
    downloaded_at_utc: str,
) -> Dict[str, Any]:
    """按  文档生成 MANIFEST.yaml 的结构。"""

    return {
        "dataset_name": config["dataset_name"],
        "source_name": config.get("source_name", "Amazon Science"),
        "source_url": config["source_url"],
        "repository_url": config["repository_url"],
        "downloaded_at_utc": downloaded_at_utc,
        "license_status": config.get("license_status", "verify-before-redistribution"),
        "raw_dir": config["raw_dir"],
        "raw_files": files,
        "processing": {
            "schema_version": "",
            "locale_scope": config.get("locale_scope", "all"),
            "version_scope": config.get("version_scope", "all"),
        },
    }


def run(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    project_root = config_project_root(config_path)
    raw_dir = resolve_project_path(config["raw_dir"], project_root)
    manifest_path = resolve_project_path(config["manifest_path"], project_root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_manifest = _load_existing_manifest(manifest_path)
    metadata: List[Dict[str, Any]] = []
    downloaded_any = False

    for spec in file_specs(config):
        name = spec["name"]
        url = spec["url"]
        destination = raw_dir / name
        configured_hash = spec.get("sha256")
        previous_hash = _manifest_hash(existing_manifest, name)
        expected_hash = configured_hash or previous_hash

        if destination.exists():
            if destination.stat().st_size <= 0:
                raise DownloadError(
                    "发现已有空文件，为避免覆盖原始数据已停止，请人工处理: %s" % destination
                )
            print("复用已有原始文件: %s" % destination)
        else:
            print("开始下载: %s" % name)
            _download_to_file(url, destination)
            downloaded_any = True
            print("下载完成: %s" % destination)

        metadata.append(_file_metadata(destination, name, url, expected_hash))

    previous_downloaded_at = existing_manifest.get("downloaded_at_utc")
    if not downloaded_any and isinstance(previous_downloaded_at, str) and previous_downloaded_at:
        downloaded_at_utc = previous_downloaded_at
    else:
        downloaded_at_utc = utc_now()

    manifest = build_manifest(config, metadata, downloaded_at_utc)
    write_yaml_atomic(manifest_path, manifest)
    print("MANIFEST 已写入: %s" % manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Amazon ESCI  原始数据")
    parser.add_argument("--config", required=True, type=Path, help="data.yaml 路径")
    args = parser.parse_args()
    try:
        run(args.config)
    except (KeyError, ValueError, DownloadError, OSError) as exc:
        print("错误: %s" % exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
