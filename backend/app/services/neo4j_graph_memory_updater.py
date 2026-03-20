"""
Neo4j Graph Memory Updater
更新 Neo4j 图数据库中的记忆信息
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from neo4j import GraphDatabase

from ..config import Config


@dataclass
class AgentActivity:
    """智能体活动"""
    agent_id: str
    agent_name: str
    action: str
    content: str
    timestamp: datetime
    round_number: int


class Neo4jGraphMemoryManager:
    """
    管理 Neo4j 中的记忆
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri or Config.NEO4J_URI
        self.username = username or Config.NEO4J_USERNAME
        self.password = password or Config.NEO4J_PASSWORD
        self.database = Config.NEO4J_DATABASE

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def close(self):
        if self.driver:
            self.driver.close()

    def get_context(self, graph_id: str, query: str, limit: int = 10) -> str:
        """
        获取相关上下文

        Args:
            graph_id: 图谱ID
            query: 查询关键词
            limit: 返回结果数量

        Returns:
            上下文字符串
        """
        with self.driver.session(database=self.database) as session:
            # 搜索相关实体
            query_cypher = """
                MATCH (n:Entity {graph_id: $graph_id})
                WHERE toLower(n.name) CONTAINS toLower($query)
                   OR toLower(n.description) CONTAINS toLower($query)
                RETURN n.name AS name, n.type AS type, n.description AS description
                LIMIT $limit
            """
            result = session.run(query_cypher, graph_id=graph_id, query=query, limit=limit)

            context_parts = []
            for record in result:
                name = record["name"]
                entity_type = record["type"]
                description = record.get("description") or ""

                context_parts.append(f"- {name} ({entity_type}): {description}")

            return "\n".join(context_parts) if context_parts else "No relevant context found."

    def add_episode(self, graph_id: str, content: str, episode_type: str = "text") -> str:
        """添加新的记忆片段"""
        import uuid
        episode_uuid = str(uuid.uuid4())

        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (p:Project {graph_id: $graph_id})
                CREATE (e:Episode {
                    uuid: $episode_uuid,
                    content: $content,
                    type: $episode_type,
                    graph_id: $graph_id,
                    created_at: datetime()
                })
                CREATE (p)-[:HAS_EPISODE]->(e)
                """,
                graph_id=graph_id,
                episode_uuid=episode_uuid,
                content=content,
                episode_type=episode_type
            )

        return episode_uuid

    def link_entity_to_episode(self, graph_id: str, entity_name: str, episode_uuid: str):
        """将实体关联到记忆片段"""
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (n:Entity {graph_id: $graph_id, name: $entity_name})
                MATCH (e:Episode {graph_id: $graph_id, uuid: $episode_uuid})
                MERGE (n)-[:MENTIONED_IN]->(e)
                """,
                graph_id=graph_id,
                entity_name=entity_name,
                episode_uuid=episode_uuid
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class Neo4jGraphMemoryUpdater:
    """
    更新 Neo4j 图数据库中的记忆
    """

    def __init__(self, graph_id: str, uri: str = None, username: str = None, password: str = None):
        self.graph_id = graph_id
        self.manager = Neo4jGraphMemoryManager(uri, username, password)

    def close(self):
        self.manager.close()

    def update_memory(self, activities: List[AgentActivity]) -> bool:
        """
        更新记忆

        Args:
            activities: 智能体活动列表

        Returns:
            是否成功
        """
        try:
            for activity in activities:
                # 创建记忆片段
                content = f"[Round {activity.round_number}] {activity.agent_name}: {activity.action} - {activity.content}"
                episode_uuid = self.manager.add_episode(
                    self.graph_id,
                    content,
                    episode_type=f"agent_{activity.action}"
                )

                # 关联到智能体实体
                self.manager.link_entity_to_episode(
                    self.graph_id,
                    activity.agent_name,
                    episode_uuid
                )

            return True
        except Exception as e:
            print(f"Memory update error: {e}")
            return False

    def get_simulation_context(self, query: str, limit: int = 10) -> str:
        """获取模拟上下文"""
        return self.manager.get_context(self.graph_id, query, limit)

    def add_entity(self, name: str, entity_type: str, description: str = ""):
        """添加实体"""
        with self.manager.driver.session(database=self.manager.database) as session:
            session.run(
                """
                MERGE (n:Entity {graph_id: $graph_id, name: $name})
                SET n.type = $entity_type,
                    n.description = $description,
                    n.created_at = datetime()
                """,
                graph_id=self.graph_id,
                name=name,
                entity_type=entity_type,
                description=description
            )

    def add_relation(self, source: str, target: str, relation_type: str, description: str = ""):
        """添加关系"""
        with self.manager.driver.session(database=self.manager.database) as session:
            safe_type = relation_type.replace(" ", "_").replace("-", "_")
            session.run(
                f"""
                MATCH (s:Entity {{graph_id: $graph_id, name: $source}})
                MATCH (t:Entity {{graph_id: $graph_id, name: $target}})
                MERGE (s)-[r:`{safe_type}`]->(t)
                SET r.description = $description,
                    r.created_at = datetime()
                """,
                graph_id=self.graph_id,
                source=source,
                target=target,
                description=description
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
