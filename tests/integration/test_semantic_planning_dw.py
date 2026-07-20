import asyncio
from types import MappingProxyType

from sqlalchemy import text

from app.agent.semantic_planning.catalog import (
    ColumnCandidate,
    SemanticCandidateCatalog,
    ValueCandidate,
)
from app.agent.semantic_planning.draft import EnumPredicateMention
from app.agent.semantic_planning.resolvers.enum_predicate import (
    EnumResolutionContext,
    resolve_enum_predicate,
)
from app.clients.mysql_client_manager import MySQLClientManager
from app.conf.app_config import app_config
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository


def _catalog(column_id, canonical_value):
    table, column = column_id.split(".", 1)
    candidate = ColumnCandidate(
        candidate_id=column_id,
        table=table,
        name=column,
        aliases=(),
        role="dimension",
        projectable=True,
        data_type="varchar",
    )
    return SemanticCandidateCatalog(
        metadata_version="integration",
        tables=MappingProxyType({}),
        columns=MappingProxyType({column_id: candidate}),
        relationships=MappingProxyType({}),
        metrics=MappingProxyType({}),
        values=MappingProxyType(
            {
                "meta-alias-value": ValueCandidate(
                    candidate_id="meta-alias-value",
                    canonical_value=canonical_value,
                    aliases=(),
                    column_id=column_id,
                    source="meta_alias",
                )
            }
        ),
    )


async def _resolve(raw_text, column_id, repository):
    mention = EnumPredicateMention(
        raw_text=raw_text,
        value_candidate_ids=["meta-alias-value"],
        operator_intent="eq",
    )
    return await resolve_enum_predicate(
        mention,
        EnumResolutionContext(
            catalog=_catalog(column_id, raw_text),
            dw_repository=repository,
            trusted_sources=(f"统计{raw_text}销售额",),
        ),
    )


def test_meta_alias_candidate_uses_exact_dw_verification():
    async def run():
        manager = MySQLClientManager(app_config.db_dw)
        manager.init()
        try:
            async with manager.session_factory() as session:
                result = await _resolve(
                    "华北",
                    "dim_region.region_name",
                    DWMySQLRepository(session),
                )
                assert result.status == "resolved"
                assert result.plan.canonical_values == ["华北"]
        finally:
            await manager.close()

    asyncio.run(run())


def test_meta_alias_candidate_does_not_fuzzy_match_database_value():
    async def run():
        manager = MySQLClientManager(app_config.db_dw)
        manager.init()
        try:
            async with manager.session_factory() as session:
                await session.execute(
                    text(
                        "CREATE TEMPORARY TABLE tmp_semantic_region "
                        "(region_name VARCHAR(64) NOT NULL)"
                    )
                )
                await session.execute(
                    text(
                        "INSERT INTO tmp_semantic_region(region_name) "
                        "VALUES ('华北地区')"
                    )
                )
                try:
                    result = await _resolve(
                        "华北",
                        "tmp_semantic_region.region_name",
                        DWMySQLRepository(session),
                    )
                    assert result.status == "unresolved"
                    assert result.issue.code == "value_not_found"
                finally:
                    await session.execute(text("DROP TEMPORARY TABLE tmp_semantic_region"))
        finally:
            await manager.close()

    asyncio.run(run())
