SET NAMES utf8mb4;
CREATE DATABASE meta DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
GRANT ALL PRIVILEGES ON meta.* TO 'cr730'@'%';

USE meta;

DROP TABLE IF EXISTS table_info;
CREATE TABLE table_info
(
    id          VARCHAR(64) PRIMARY KEY COMMENT '表编号',
    name        VARCHAR(128) COMMENT '表名称',
    role        VARCHAR(32) COMMENT '表类型(fact/dim)',
    description TEXT COMMENT '表描述'
);



DROP TABLE IF EXISTS column_info;
CREATE TABLE column_info
(
    id          VARCHAR(64) PRIMARY KEY COMMENT '列编号',
    name        VARCHAR(128) COMMENT '列名称',
    type        VARCHAR(64) COMMENT '数据类型',
    role        VARCHAR(32) COMMENT '列类型(primary_key,foreign_key,measure,dimension)',
    examples    JSON COMMENT '数据示例',
    description TEXT COMMENT '列描述',
    alias       JSON COMMENT '列别名',
    table_id    VARCHAR(64) COMMENT '所属表编号'
);

DROP TABLE IF EXISTS metric_info;
CREATE TABLE metric_info
(
    id               VARCHAR(64) PRIMARY KEY COMMENT '指标编码',
    name             VARCHAR(128) COMMENT '指标名称',
    description      TEXT COMMENT '指标描述',
    relevant_columns JSON COMMENT '关联的列',
    alias            JSON COMMENT '指标别名'
);


DROP TABLE IF EXISTS column_metric;
CREATE TABLE column_metric
(
    column_id VARCHAR(64) COMMENT '列编号',
    metric_id VARCHAR(64) COMMENT '指标编号',
    PRIMARY KEY (column_id, metric_id)
);

DROP TABLE IF EXISTS metadata_build;
CREATE TABLE metadata_build
(
    id          BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '构建记录编号',
    version     VARCHAR(64)  NOT NULL COMMENT '元数据构建版本',
    config_path VARCHAR(512) NOT NULL COMMENT '构建配置文件路径',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '构建完成时间'
);

DROP TABLE IF EXISTS conversation_snapshot;
DROP TABLE IF EXISTS conversation_turn;
DROP TABLE IF EXISTS conversation;

CREATE TABLE conversation
(
    id          VARCHAR(64) PRIMARY KEY COMMENT '会话编号',
    user_id     VARCHAR(128) NULL COMMENT '用户编号',
    title       VARCHAR(255) NOT NULL COMMENT '会话标题',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    archived_at TIMESTAMP NULL COMMENT '归档时间'
);

CREATE TABLE conversation_turn
(
    id                   VARCHAR(64) PRIMARY KEY COMMENT '轮次编号',
    conversation_id      VARCHAR(64) NOT NULL COMMENT '会话编号',
    turn_index           INT NOT NULL COMMENT '会话内轮次序号',
    user_query           TEXT NOT NULL COMMENT '用户原始问题',
    rewritten_query      TEXT NOT NULL COMMENT '追问改写后的完整问题',
    sql_text             TEXT NULL COMMENT '本轮生成或修正后的 SQL',
    final_answer_summary TEXT NULL COMMENT '查询结果摘要',
    safety_error         TEXT NULL COMMENT '安全或业务语义错误',
    blocked_by           VARCHAR(64) NULL COMMENT '拦截节点',
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_conversation_turn_conversation (conversation_id, turn_index)
);

CREATE TABLE conversation_snapshot
(
    conversation_id       VARCHAR(64) PRIMARY KEY COMMENT '会话编号',
    last_metric_bindings  JSON NULL COMMENT '上一轮指标绑定快照',
    last_resolved_filters JSON NULL COMMENT '上一轮过滤条件绑定快照',
    last_time_binding     JSON NULL COMMENT '上一轮时间绑定快照',
    last_sql              TEXT NULL COMMENT '上一轮 SQL',
    last_answer_summary   TEXT NULL COMMENT '上一轮结果摘要',
    recent_turns_summary  JSON NULL COMMENT '最近轮次摘要',
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'
);
