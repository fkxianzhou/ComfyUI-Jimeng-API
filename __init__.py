import sys
import subprocess
import importlib.metadata
import locale
from .constants import LOG_TRANSLATIONS

# V3 API 引入
from comfy_api.latest import ComfyExtension

# 导入所有迁移后的节点类
from .nodes_shared import JimengAPIClient
from .nodes_image import JimengSeedream3, JimengSeedream4
from .nodes_video import JimengSeedance1, JimengSeedance1_5,JimengReferenceImage2Video, JimengVideoQueryTasks

def get_init_text(key, **kwargs):
    lang_code = "en"
    try:
        sys_lang, _ = locale.getdefaultlocale()
        if sys_lang and sys_lang.startswith("zh"):
            lang_code = "zh"
    except:
        pass

    mapping = LOG_TRANSLATIONS.get(lang_code, LOG_TRANSLATIONS["en"])
    msg = mapping.get(key, LOG_TRANSLATIONS["en"].get(key, key))

    try:
        return msg.format(**kwargs)
    except:
        return msg


def check_and_update_dependencies():
    package_name = "volcengine-python-sdk"
    install_spec = "volcengine-python-sdk[ark]>=4.0.43"
    min_version = "4.0.43"

    try:
        current_version = importlib.metadata.version(package_name)

        try:
            from packaging import version

            if version.parse(current_version) < version.parse(min_version):
                print(
                    get_init_text(
                        "init_sdk_ver_low", current=current_version, min=min_version
                    )
                )

                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", install_spec]
                )
                print(get_init_text("init_sdk_update_ok"))
        except ImportError:
            pass

    except importlib.metadata.PackageNotFoundError:
        print(get_init_text("init_sdk_not_found", pkg=package_name))
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_spec]
            )
            print(get_init_text("init_sdk_install_ok"))
        except Exception as e:
            print(get_init_text("init_sdk_install_fail", e=e))
    except Exception as e:
        print(get_init_text("init_dep_check_err", e=e))

# 执行依赖检查
check_and_update_dependencies()

# V3 扩展注册
class JimengExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [
            JimengAPIClient,
            JimengSeedream3,
            JimengSeedream4,
            JimengSeedance1,
            JimengSeedance1_5,
            JimengReferenceImage2Video,
            JimengVideoQueryTasks,
        ]

async def comfy_entrypoint() -> ComfyExtension:
    return JimengExtension()

# 保留 Web 目录配置,供前端资源加载使用
WEB_DIRECTORY = "./web"
__all__ = ["WEB_DIRECTORY"]