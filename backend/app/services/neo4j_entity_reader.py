"""
Neo4j Entity Reader
从 Neo4j 图数据库读取实体信息
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from neo4j import GraphDatabase

from ..config import Config


@dataclass
class EntityNode:
    """实体节点"""
    uuid: str
    name: str
    entity_type: str
    description: str = ""
    summary: str = ""
    attributes: Dict[str, Any] = None
    related_edges: List[Dict[str, Any]] = None
    related_nodes: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}
        if self.related_edges is None:
            self.related_edges = []
        if self.related_nodes is None:
            self.related_nodes = []

    def get_entity_type(self) -> str:
        """Backward compatibility method"""
        return self.entity_type

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "uuid": self.uuid,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }


@dataclass
class FilteredEntities:
    """过滤后的实体"""
    entities: List[EntityNode]
    query: str = ""
    entity_types: set = None
    total_count: int = 0
    filtered_count: int = 0

    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = set()
        if self.total_count == 0:
            self.total_count = len(self.entities)
        if self.filtered_count == 0:
            self.filtered_count = len(self.entities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class Neo4jEntityReader:
    """
    从 Neo4j 图数据库读取实体
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
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def get_all_nodes(self, graph_id: str, entity_type: str = None) -> List[EntityNode]:
        """
        获取所有实体节点

        Args:
            graph_id: 图谱ID
            entity_type: 实体类型过滤

        Returns:
            实体节点列表
        """
        with self.driver.session(database=self.database) as session:
            if entity_type:
                query = """
                    MATCH (n:Entity {graph_id: $graph_id})
                    WHERE n.type = $entity_type
                    RETURN n.uuid AS uuid, n.name AS name, n.type AS entity_type,
                           n.description AS description, n.summary AS summary,
                           n.attributes AS attributes
                """
                result = session.run(query, graph_id=graph_id, entity_type=entity_type)
            else:
                query = """
                    MATCH (n:Entity {graph_id: $graph_id})
                    RETURN n.uuid AS uuid, n.name AS name, n.type AS entity_type,
                           n.description AS description, n.summary AS summary,
                           n.attributes AS attributes
                """
                result = session.run(query, graph_id=graph_id)

            nodes = []
            for record in result:
                nodes.append(EntityNode(
                    uuid=record["uuid"] or "",
                    name=record["name"] or "",
                    entity_type=record["entity_type"] or "Entity",
                    description=record.get("description") or "",
                    summary=record.get("summary") or "",
                    attributes=dict(record.get("attributes") or {})
                ))
            return nodes

    def get_node_by_name(self, graph_id: str, name: str) -> Optional[EntityNode]:
        """根据名称获取实体"""
        with self.driver.session(database=self.database) as session:
            query = """
                MATCH (n:Entity {graph_id: $graph_id, name: $name})
                RETURN n.uuid AS uuid, n.name AS name, n.type AS entity_type,
                       n.description AS description, n.summary AS summary,
                       n.attributes AS attributes
            """
            result = session.run(query, graph_id=graph_id, name=name)
            record = result.single()

            if record:
                return EntityNode(
                    uuid=record["uuid"] or "",
                    name=record["name"] or "",
                    entity_type=record["entity_type"] or "Entity",
                    description=record.get("description") or "",
                    summary=record.get("summary") or "",
                    attributes=dict(record.get("attributes") or {})
                )
            return None

    def get_node_edges(self, graph_id: str, node_name: str) -> List[Dict]:
        """获取实体的所有关系"""
        with self.driver.session(database=self.database) as session:
            query = """
                MATCH (n:Entity {graph_id: $graph_id, name: $node_name})-[r]-(m)
                RETURN type(r) AS relation_type,
                       startNode(r).name AS source_name,
                       endNode(r).name AS target_name,
                       r.description AS description
            """
            result = session.run(query, graph_id=graph_id, node_name=node_name)

            edges = []
            for record in result:
                edges.append({
                    "type": record["relation_type"],
                    "source": record["source_name"],
                    "target": record["target_name"],
                    "description": record.get("description") or ""
                })
            return edges

    def search_nodes(self, graph_id: str, keyword: str) -> List[EntityNode]:
        """搜索实体"""
        with self.driver.session(database=self.database) as session:
            query = """
                MATCH (n:Entity {graph_id: $graph_id})
                WHERE toLower(n.name) CONTAINS toLower($keyword)
                   OR toLower(n.description) CONTAINS toLower($keyword)
                RETURN n.uuid AS uuid, n.name AS name, n.type AS entity_type,
                       n.description AS description, n.summary AS summary,
                       n.attributes AS attributes
                LIMIT 50
            """
            result = session.run(query, graph_id=graph_id, keyword=keyword)

            nodes = []
            for record in result:
                nodes.append(EntityNode(
                    uuid=record["uuid"] or "",
                    name=record["name"] or "",
                    entity_type=record["entity_type"] or "Entity",
                    description=record.get("description") or "",
                    summary=record.get("summary") or "",
                    attributes=dict(record.get("attributes") or {})
                ))
            return nodes

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> FilteredEntities:
        """按类型获取实体"""
        entities = self.get_all_nodes(graph_id, entity_type)
        return FilteredEntities(
            entities=entities,
            query=f"type:{entity_type}"
        )

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        过滤图谱节点，只返回有自定义类型的节点。

        Args:
            graph_id: 图谱ID
            defined_entity_types: 可选的实体类型白名单
            enrich_with_edges: 是否填充边信息

        Returns:
            FilteredEntities 包含匹配的实体和统计信息
        """
        import uuid as uuid_module

        # 获取所有节点
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        # 过滤节点
        filtered_entities = []
        entity_types_found = set()

        for node in all_nodes:
            # 跳过没有自定义类型的节点
            if node.entity_type in ["Entity", "Node", ""]:
                continue

            # 如果提供了白名单，检查是否匹配
            if defined_entity_types:
                if node.entity_type not in defined_entity_types:
                    continue

            entity_types_found.add(node.entity_type)

            # 如果需要边信息，获取
            if enrich_with_edges:
                node.related_edges = self.get_node_edges(graph_id, node.name)

            filtered_entities.append(node)

        return FilteredEntities(
            entities=filtered_entities,
            query=""
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """获取单个实体的完整上下文"""
        # 通过名称查找
        all_nodes = self.get_all_nodes(graph_id)
        for node in all_nodes:
            if node.uuid == entity_uuid:
                node.related_edges = self.get_node_edges(graph_id, node.name)
                return node
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
