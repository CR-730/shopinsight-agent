"""元数据知识库后台轮询调度。"""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.core.log import logger

BuildCallable = Callable[[Path], Awaitable[None]]


class MetaKnowledgeScheduler:
    """轮询元数据配置文件变化，变化时触发知识库重建。"""

    def __init__(
        self,
        config_path: Path,
        poll_interval_seconds: int,
        build: BuildCallable,
        build_on_start: bool,
    ):
        self.config_path = config_path
        self.poll_interval_seconds = poll_interval_seconds
        self.build = build
        self.build_on_start = build_on_start
        self._last_signature: str | None = None
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._build_lock = asyncio.Lock()

    async def poll_once(self):
        """执行一次轮询检查，便于测试和手动触发。"""

        if not self.config_path.exists():
            logger.warning(f"元数据配置文件不存在，跳过轮询：{self.config_path}")
            return

        signature = self._file_signature(self.config_path)
        if self._last_signature == signature:
            return
        if self._last_signature is None and not self.build_on_start:
            self._last_signature = signature
            return

        async with self._build_lock:
            latest_signature = self._file_signature(self.config_path)
            if self._last_signature == latest_signature:
                return
            await self.build(self.config_path)
            self._last_signature = latest_signature

    def start(self):
        """在当前事件循环中启动后台轮询任务。"""

        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        """停止后台轮询任务。"""

        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self):
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except Exception as exc:
                logger.exception(f"元数据后台构建失败：{exc}")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval_seconds
                )
            except TimeoutError:
                continue

    @staticmethod
    def _file_signature(path: Path) -> str:
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        return digest.hexdigest()
