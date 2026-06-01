import asyncio

import pytest

from app.services.meta_knowledge_scheduler import MetaKnowledgeScheduler


@pytest.mark.anyio
async def test_scheduler_builds_once_on_start_when_config_exists(tmp_path):
    config_path = tmp_path / "meta_config.yaml"
    config_path.write_text("tables: []\n", encoding="utf-8")
    builds = []

    async def build(path):
        builds.append(path)

    scheduler = MetaKnowledgeScheduler(
        config_path=config_path,
        poll_interval_seconds=60,
        build=build,
        build_on_start=True,
    )

    await scheduler.poll_once()

    assert builds == [config_path]


@pytest.mark.anyio
async def test_scheduler_skips_when_file_signature_is_unchanged(tmp_path):
    config_path = tmp_path / "meta_config.yaml"
    config_path.write_text("tables: []\n", encoding="utf-8")
    builds = []

    async def build(path):
        builds.append(path)

    scheduler = MetaKnowledgeScheduler(
        config_path=config_path,
        poll_interval_seconds=60,
        build=build,
        build_on_start=True,
    )

    await scheduler.poll_once()
    await scheduler.poll_once()

    assert builds == [config_path]


@pytest.mark.anyio
async def test_scheduler_rebuilds_when_file_changes(tmp_path):
    config_path = tmp_path / "meta_config.yaml"
    config_path.write_text("tables: []\n", encoding="utf-8")
    builds = []

    async def build(path):
        builds.append(path)

    scheduler = MetaKnowledgeScheduler(
        config_path=config_path,
        poll_interval_seconds=60,
        build=build,
        build_on_start=True,
    )

    await scheduler.poll_once()
    await asyncio.sleep(0)
    config_path.write_text("metrics: []\n", encoding="utf-8")
    await scheduler.poll_once()

    assert builds == [config_path, config_path]
