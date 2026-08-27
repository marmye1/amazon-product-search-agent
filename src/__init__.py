"""亚马逊商品检索项目包初始化。"""

import warnings


# 当前 macOS Python 3.9 使用 LibreSSL；该兼容性提示不代表请求失败，
# 只在项目内隐藏这条已知启动噪声，真实请求异常仍会正常抛出。
warnings.filterwarnings(
    "ignore",
    message=r"urllib3  only supports OpenSSL.*",
    category=Warning,
)
