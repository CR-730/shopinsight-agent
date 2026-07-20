"""
数仓 MySQL 仓储

这一层对应文档里的 DW Repository，职责是到真实数仓中补齐配置文件里
没有显式维护的信息，例如字段类型和字段示例值。Service 层只关心
“需要哪些信息”，具体怎样查数仓由仓储层统一封装
SQL 生成闭环中的数据库环境读取 SQL 校验和最终查询执行也集中放在这里
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class DWMySQLRepository:
    """负责查询数仓真实表结构和字段样例值"""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._column_value_exists_cache: dict[tuple[str, str, str], bool] = {}
        self._db_info_cache: dict | None = None

    async def get_column_types(self, table_name: str) -> dict[str, str]:
        """查询整张表的字段类型，作为 ColumnInfo.type 的真实来源"""
        sql = f"show columns from {table_name}"
        result = await self.session.execute(text(sql))
        result_dict = result.mappings().fetchall()
        return {row["Field"]: row["Type"] for row in result_dict}

    async def get_column_values(
        self, table_name: str, column_name: str, limit: int = 10
    ) -> list:
        """抽样查询字段示例值，供元数据入库和后续检索链路复用"""
        sql = f"select distinct {column_name} from {table_name} limit {limit}"
        result = await self.session.execute(text(sql))
        return [row[0] for row in result.fetchall()]

    async def column_value_exists(
        self, table_name: str, column_name: str, value: str
    ) -> bool:
        """验证元数据别名对应的规范枚举值在真实数仓中存在。"""

        cache_key = (table_name, column_name, value)
        if cache_key in self._column_value_exists_cache:
            return self._column_value_exists_cache[cache_key]

        sql = (
            f"select 1 from {table_name} "
            f"where {column_name} = :value limit 1"
        )
        result = await self.session.execute(text(sql), {"value": value})
        exists = result.first() is not None
        self._column_value_exists_cache[cache_key] = exists
        return exists

    async def get_db_info(self):
        """读取当前数仓数据库的方言和版本，供 SQL 生成提示词使用"""

        if self._db_info_cache is not None:
            return self._db_info_cache

        sql = "select version()"
        result = await self.session.execute(text(sql))
        version = result.scalar()

        # dialect 来自 SQLAlchemy 当前绑定的数据库方言，例如 mysql
        dialect = self.session.bind.dialect.name
        self._db_info_cache = {"dialect": dialect, "version": version}
        return self._db_info_cache

    async def validate(self, sql: str):
        """用 EXPLAIN 让数据库提前解析 SQL，发现语法 表名 字段名等错误"""
        sql = f"explain {sql}"
        await self.session.execute(text(sql))

    async def run(self, sql: str) -> list[dict]:
        """执行最终 SQL，并把 SQLAlchemy 行对象转换成前端更易消费的字典列表"""
        result = await self.session.execute(text(sql))
        return [dict(row) for row in result.mappings().fetchall()]
