import sys
import traceback
import logging
import subprocess
import importlib.metadata
import locale
import asyncio
from .nodes.constants import LOG_TRANSLATIONS

_original_print_exception = traceback.print_exception
_original_format_exception = traceback.format_exception
_original_logging_error = logging.error

def _jimeng_print_exception(*args, **kwargs):
    """
    自定义异常打印函数。
    如果是自定义异常且标记了 suppress_traceback，则只打印错误信息，不打印堆栈。
    """
    exc = None
    if len(args) > 0:
        if isinstance(args[0], BaseException):
            exc = args[0]
        elif isinstance(args[0], type) and issubclass(args[0], BaseException) and len(args) > 1:
            exc = args[1]

    if exc and getattr(exc, "jimeng_suppress_traceback", False):
        f = kwargs.get('file')
        if not f:
            f = sys.stderr
        
        print(f"{exc}", file=f)
        return

    return _original_print_exception(*args, **kwargs)

def _jimeng_format_exception(*args, **kwargs):
    """
    自定义异常格式化函数。
    如果是自定义异常且标记了 suppress_traceback，则只返回异常信息字符串。
    """
    exc = None
    if len(args) >= 2:
        exc = args[1]
    
    if exc and getattr(exc, "jimeng_suppress_traceback", False):
        return [f"{exc}\n"]
    
    return _original_format_exception(*args, **kwargs)

def _jimeng_logging_error(msg, *args, **kwargs):
    """
    自定义日志记录错误函数。
    如果遇到 "!!! Exception during processing !!!" 且是自定义异常，则抑制该日志。
    """
    if isinstance(msg, str) and msg.startswith("!!! Exception during processing !!!"):
        exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_value and getattr(exc_value, "jimeng_suppress_traceback", False):
            return

    return _original_logging_error(msg, *args, **kwargs)

traceback.print_exception = _jimeng_print_exception
traceback.format_exception = _jimeng_format_exception
logging.error = _jimeng_logging_error

if sys.platform == 'win32':
    try:
        from asyncio import proactor_events
        
        _original_call_connection_lost = proactor_events._ProactorBasePipeTransport._call_connection_lost
        
        def _silenced_call_connection_lost(self, exc):
            """
            在 Windows 上抑制 asyncio 的 ConnectionResetError 和特定的 OSError (winerror 10054)。
            """
            try:
                _original_call_connection_lost(self, exc)
            except ConnectionResetError:
                pass
            except OSError as e:
                if getattr(e, 'winerror', None) == 10054:
                    pass
                else:
                    raise

        proactor_events._ProactorBasePipeTransport._call_connection_lost = _silenced_call_connection_lost
    except ImportError:
        pass

from comfy_api.latest import ComfyExtension

from .nodes.nodes_shared import JimengAPIClient
from .nodes.nodes_image import JimengSeedream3, JimengSeedream4
from .nodes.nodes_video import JimengSeedance1, JimengSeedance1_5,JimengReferenceImage2Video, JimengVideoQueryTasks
from .nodes.quota import JimengQuotaSettings

def get_init_text(key, **kwargs):
    """
    获取初始化过程中的本地化文本。
    根据系统语言自动选择中文或英文。
    """
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
    """
    检查并自动更新依赖项 (volcengine-python-sdk)。
    如果未安装或版本过低，将尝试自动安装/更新。
    """
    package_name = "volcengine-python-sdk"
    install_spec = "volcengine-python-sdk[ark]>=5.0.6"
    min_version = "5.0.6"

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

check_and_update_dependencies()

class JimengExtension(ComfyExtension):
    """
    Jimeng 插件扩展类，用于注册节点。
    """
    async def get_node_list(self) -> list[type]:
        return [
            JimengAPIClient,
            JimengSeedream3,
            JimengSeedream4,
            JimengSeedance1,
            JimengSeedance1_5,
            JimengReferenceImage2Video,
            JimengVideoQueryTasks,
            JimengQuotaSettings,
        ]

async def comfy_entrypoint() -> ComfyExtension:
    """
    ComfyUI 插件入口点。
    """
    return JimengExtension()


WEB_DIRECTORY = "./web"
__all__ = ["WEB_DIRECTORY"]