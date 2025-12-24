import os
import io
import asyncio
import aiohttp
import torch
import numpy
import PIL.Image
import folder_paths
import random
import shutil
from .nodes_shared import log_msg

DEFAULT_DOWNLOAD_TIMEOUT = 60
DEFAULT_DOWNLOAD_RETRIES = 3


async def _fetch_data_from_url_async(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
    retries: int = DEFAULT_DOWNLOAD_RETRIES,
) -> bytes:
    for attempt in range(1, retries + 2):
        try:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.get(url, timeout=client_timeout) as response:
                response.raise_for_status()
                return await response.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt > retries:
                raise e
            retry_delay = 2
            log_msg(
                "download_retry",
                attempt=attempt,
                total=retries + 1,
                delay=retry_delay,
                e=e,
            )
            await asyncio.sleep(retry_delay)
    return b""


async def download_url_to_image_tensor_async(
    session: aiohttp.ClientSession, url: str
) -> torch.Tensor | None:
    if not url:
        return None
    try:
        image_data = await _fetch_data_from_url_async(session, url)
        i = PIL.Image.open(io.BytesIO(image_data))
        image = i.convert("RGB")
        image = numpy.array(image).astype(numpy.float32) / 255.0
        return torch.from_numpy(image)[None,]
    except Exception as e:
        log_msg("err_download_url", url=url, e=e)
        return None


async def _download_to_temp_base(
    session: aiohttp.ClientSession,
    url: str,
    prefix: str,
    seed: int | None,
    save_path_name: str,
    file_ext: str,
) -> tuple[str | None, bytes | None]:
    if not url:
        return (None, None)

    if seed is not None:
        prefix = f"{prefix}_seed_{seed}"

    output_dir = folder_paths.get_temp_directory()
    if save_path_name:
        output_dir = os.path.join(output_dir, save_path_name)

    (full_output_folder, filename, _, _, _) = folder_paths.get_save_image_path(
        prefix, output_dir
    )
    os.makedirs(full_output_folder, exist_ok=True)

    final_filename = f"{filename}_{random.randint(1, 10000)}.{file_ext}"
    final_path = os.path.join(full_output_folder, final_filename)

    try:
        data = await _fetch_data_from_url_async(session, url)
        with open(final_path, "wb") as f:
            f.write(data)
        return (final_path, data)
    except Exception as e:
        log_msg("err_download_url", url=url, e=e)
        return (None, None)


async def download_video_to_temp(
    session: aiohttp.ClientSession,
    url: str,
    prefix: str,
    seed: int | None,
    save_path_name: str,
) -> str | None:
    if not url:
        return None
    file_ext = url.split(".")[-1].split("?")[0] or "mp4"
    (path, _) = await _download_to_temp_base(
        session, url, prefix, seed, save_path_name, file_ext
    )
    return path


async def download_image_to_temp(
    session: aiohttp.ClientSession,
    url: str,
    prefix: str,
    seed: int | None,
    save_path_name: str,
) -> tuple[torch.Tensor | None, str | None]:
    if not url:
        return (None, None)

    file_ext = url.split(".")[-1].split("?")[0] or "jpg"
    (path, data) = await _download_to_temp_base(
        session, url, prefix, seed, save_path_name, file_ext
    )

    tensor = None
    if path and data:
        try:
            i = PIL.Image.open(io.BytesIO(data))
            image = i.convert("RGB")
            image = numpy.array(image).astype(numpy.float32) / 255.0
            tensor = torch.from_numpy(image)[None,]
        except Exception as e:
            log_msg("err_convert_tensor", e=e)

    return (tensor, path)


def save_to_output(src_path: str, filename_prefix: str):
    if not src_path or not os.path.exists(src_path):
        return

    try:
        output_dir = folder_paths.get_output_directory()
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir)
        
        ext = os.path.splitext(src_path)[1]
        if not ext:
            ext = ".mp4"

        dest_filename = f"{filename}_{counter:05}_{ext}"
        dest_path = os.path.join(full_output_folder, dest_filename)

        shutil.copy2(src_path, dest_path)
    except Exception as e:
        log_msg("err_copy_fail", path=src_path, e=e)