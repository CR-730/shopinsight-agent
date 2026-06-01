"""
元数据库 MySQL 仓储

这一层对应文档里的 Meta Repository，负责接收业务实体并落到 Meta MySQL
Repository 自身只关心“如何写入”，而“哪些写操作要放在同一笔事务里”，由 Service 层统一决定

表 字段 指标和字段指标关系都会先以业务实体流转，再在这里统一转成 ORM 模型
问数链路运行时也会从这里读取元数据，用来把召回到的 id 补齐成完整实体
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.entities.column_info import ColumnInfo
from app.entities.column_metric import ColumnMetric
from app.entities.metric_info import MetricInfo
from app.entities.table_info import TableInfo
from app.models.column_info import ColumnInfoMySQL
from app.models.metric_info import MetricInfoMySQL
from app.models.table_info import TableInfoMySQL
from app.repositories.mysql.meta.mappers.column_info_mapper import ColumnInfoMapper
from app.repositories.mysql.meta.mappers.column_metric_mapper import ColumnMetricMapper
from app.repositories.mysql.meta.mappers.metric_info_mapper import MetricInfoMapper
from app.repositories.mysql.meta.mappers.table_info_mapper import TableInfoMapper


class MetaMySQLRepository:
    """负责把元数据业务实体持久化到 Meta MySQL"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._metric_infos_cache: list[MetricInfo] | None = None
        self._column_infos_cache: list[ColumnInfo] | None = None

    async def clear_all(self):
        """清空元数据构建产物，让构建脚本可以安全重跑。"""
        self._metric_infos_cache = None
        self._column_infos_cache = None
        for table_name in (
            "column_metric",
            "metric_info",
            "column_info",
            "table_info",
        ):
            await self.session.execute(text(f"delete from {table_name}"))
        await self.session.commit()

    async def save_build_version(self, version: str, config_path):
        """记录一次成功构建的元数据版本，保留历史用于排查和回滚决策。"""

        await self.session.execute(
            text(
                """
                create table if not exists metadata_build
                (
                    id bigint primary key auto_increment,
                    version varchar(64) not null,
                    config_path varchar(512) not null,
                    created_at timestamp default current_timestamp
                )
                """
            )
        )
        await self.session.execute(
            text(
                "insert into metadata_build(version, config_path) "
                "values (:version, :config_path)"
            ),
            {"version": version, "config_path": str(config_path)},
        )
        await self.session.commit()

    def save_table_infos(self, table_infos: list[TableInfo]):
        """批量保存表元数据。输入仍然是业务实体，而不是 ORM 模型"""
        self.session.add_all(
            [TableInfoMapper.to_model(table_info) for table_info in table_infos]
        )

    def save_column_infos(self, column_infos: list[ColumnInfo]):
        """批量保存字段元数据。实体到模型的转换统一通过 Mapper 完成"""
        self.session.add_all(
            [ColumnInfoMapper.to_model(column_info) for column_info in column_infos]
        )

    def save_metric_infos(self, metric_infos: list[MetricInfo]):
        """批量保存指标元数据。指标本身和字段关联关系分开写入"""
        self.session.add_all(
            [MetricInfoMapper.to_model(metric_info) for metric_info in metric_infos]
        )

    def save_column_metrics(self, column_metrics: list[ColumnMetric]):
        """批量保存字段与指标的关联关系"""
        self.session.add_all(
            [
                ColumnMetricMapper.to_model(column_metric)
                for column_metric in column_metrics
            ]
        )

    async def get_column_info_by_id(self, id: str) -> ColumnInfo | None:
        """按字段 id 查询字段元数据，供召回信息合并阶段补齐字段上下文"""

        column_info: ColumnInfoMySQL | None = await self.session.get(
            ColumnInfoMySQL, id
        )
        if column_info:
            return ColumnInfoMapper.to_entity(column_info)
        else:
            return None

    async def get_table_info_by_id(self, id: str) -> TableInfo | None:
        """按表 id 查询表元数据，最终组装成提示词里的表结构信息"""

        table_info: TableInfoMySQL | None = await self.session.get(TableInfoMySQL, id)
        if table_info:
            return TableInfoMapper.to_entity(table_info)
        else:
            return None

    async def get_key_columns_by_table_id(self, table_id: str) -> list[ColumnInfo]:
        """查询指定表的主外键字段，避免 Join 关键字段被向量召回漏掉"""

        # 主外键字段用于后续生成 join 条件，不能完全依赖向量召回命中
        sql = "select * from column_info where table_id = :table_id and role in ('primary_key','foreign_key')"
        # :table_id 是 SQLAlchemy text SQL 的占位符，实际值通过第二个参数传入
        result = await self.session.execute(text(sql), {"table_id": table_id})
        # mappings() 会把结果行转成类似字典的结构，便于解包成 ColumnInfo
        return [
            ColumnInfoMapper.to_entity(ColumnInfoMySQL(**dict(row)))
            for row in result.mappings().fetchall()
        ]

    async def list_metric_infos(self) -> list[MetricInfo]:
        """查询全部指标元数据，供业务语义解析做权威匹配。"""

        if self._metric_infos_cache is not None:
            return self._metric_infos_cache

        result = await self.session.execute(text("select * from metric_info"))
        self._metric_infos_cache = [
            MetricInfoMapper.to_entity(MetricInfoMySQL(**dict(row)))
            for row in result.mappings().fetchall()
        ]
        return self._metric_infos_cache

    async def list_column_infos(self) -> list[ColumnInfo]:
        """查询全部字段元数据，供字段语义和安全策略绑定。"""

        if self._column_infos_cache is not None:
            return self._column_infos_cache

        result = await self.session.execute(text("select * from column_info"))
        self._column_infos_cache = [
            ColumnInfoMapper.to_entity(ColumnInfoMySQL(**dict(row)))
            for row in result.mappings().fetchall()
        ]
        return self._column_infos_cache
