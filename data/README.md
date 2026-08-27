# 数据目录

- `raw/esci/`：官方 ESCI 原始文件。下载后只读，不由标准化脚本覆盖。
- `staging/esci/`：数据处理临时空间。
- `processed/esci/`：按数据契约生成的标准 Parquet 文件。
- `MANIFEST.yaml`：来源、版本、文件大小和 SHA-256 清单。

原始数据不提交到 Git，也不在未确认原始数据条款前公开再分发。
