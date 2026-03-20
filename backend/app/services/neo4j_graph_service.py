"""
Neo4j Graph Service
使用 Neo4j Aura 作为图数据库，替代 Zep Cloud
"""

import os
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """Graph information."""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class Neo4jGraphService:
    """
    Neo4j Graph Service.
    使用 Neo4j Aura 构建和管理知识图谱。
    """

    def __init__(self, uri: str = None, username: str = None, password: str = None):
        self.uri = uri or Config.NEO4J_URI
        self.username = username or Config.NEO4J_USERNAME
        self.password = password or Config.NEO4J_PASSWORD
        self.database = Config.NEO4J_DATABASE
        self.task_manager = TaskManager()

        if not self.uri or not self.username or not self.password:
            raise ValueError("Neo4j connection parameters not configured")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()

    def verify_connection(self) -> bool:
        """验证连接"""
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS test")
                result.single()
            return True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False

    def create_graph(self, name: str) -> str:
        """创建图谱（实际上是创建一个项目标签）"""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

        with self.driver.session(database=self.database) as session:
            # 创建项目根节点
            session.run(
                """
                CREATE (p:Project {
                    graph_id: $graph_id,
                    name: $name,
                    created_at: datetime()
                })
                """,
                graph_id=graph_id,
                name=name
            )

        return graph_id

    def delete_graph(self, graph_id: str):
        """删除图谱"""
        with self.driver.session(database=self.database) as session:
            # 删除所有关联的节点和关系
            session.run(
                """
                MATCH (n {graph_id: $graph_id})
                DETACH DELETE n
                """,
                graph_id=graph_id
            )

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """设置本体（存储实体和关系类型定义）

        Neo4j only supports primitive types as properties, so we store as JSON strings.
        """
        import json

        entity_types = ontology.get("entity_types", [])
        edge_types = ontology.get("edge_types", [])

        with self.driver.session(database=self.database) as session:
            # 存储本体定义 - 序列化为JSON字符串以支持复杂嵌套结构
            session.run(
                """
                MATCH (p:Project {graph_id: $graph_id})
                SET p.entity_types_json = $entity_types_json,
                    p.edge_types_json = $edge_types_json,
                    p.ontology_analysis_summary = $analysis_summary
                """,
                graph_id=graph_id,
                entity_types_json=json.dumps(entity_types),
                edge_types_json=json.dumps(edge_types),
                analysis_summary=ontology.get("analysis_summary", "")
            )

    def get_ontology(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """获取本体定义"""
        import json

        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (p:Project {graph_id: $graph_id})
                RETURN p.entity_types_json AS entity_types_json,
                       p.edge_types_json AS edge_types_json,
                       p.ontology_analysis_summary AS analysis_summary
                """,
                graph_id=graph_id
            )
            record = result.single()
            if record:
                try:
                    entity_types = json.loads(record["entity_types_json"] or "[]")
                    edge_types = json.loads(record["edge_types_json"] or "[]")
                    return {
                        "entity_types": entity_types,
                        "edge_types": edge_types,
                        "analysis_summary": record["analysis_summary"] or ""
                    }
                except json.JSONDecodeError:
                    pass
        return None

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        ontology: Dict[str, Any],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """
        处理文本块：使用 LLM 提取实体和关系，然后存入 Neo4j
        返回处理的任务ID列表
        """
        episode_uuids = []
        total_chunks = len(chunks)

        # 获取实体和关系类型
        entity_types = [e["name"] for e in ontology.get("entity_types", [])]
        relation_types = [r["name"] for r in ontology.get("edge_types", [])]

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...",
                    progress,
                )

            # 为每个chunk创建episode并提取实体
            for chunk_idx, chunk in enumerate(batch_chunks):
                episode_uuid = str(uuid.uuid4())

                # 使用 LLM 提取实体和关系
                entities, relations = self._extract_entities_relations(
                    chunk,
                    entity_types,
                    relation_types
                )

                # 存储到 Neo4j
                self._store_entities_and_relations(
                    graph_id,
                    episode_uuid,
                    chunk,
                    entities,
                    relations
                )

                episode_uuids.append(episode_uuid)

            # 限制请求频率
            time.sleep(1)

        return episode_uuids

    def _extract_entities_relations(
        self,
        text: str,
        entity_types: List[str],
        relation_types: List[str]
    ) -> tuple:
        """
        使用 LLM 提取文本中的实体和关系
        返回 (entities, relations) 元组
        """
        from ..utils.llm_client import LLMClient

        llm = LLMClient()

        entity_type_str = ", ".join(entity_types) if entity_types else "Entity"
        relation_type_str = ", ".join(relation_types) if relation_types else "RELATED_TO"

        prompt = f"""从以下文本中提取实体和关系。

实体类型: {entity_type_str}
关系类型: {relation_type_str}

文本:
{text}

请以 JSON 格式返回:
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "description": "描述"}}
    ],
    "relations": [
        {{"source": "源实体", "target": "目标实体", "type": "关系类型", "description": "描述"}}
    ]
}}

只返回 JSON，不要其他内容:"""

        try:
            response = llm.call(prompt)
            import json
            # 尝试解析 JSON
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = response[start:end]
                data = json.loads(json_str)
                entities = data.get("entities", [])
                relations = data.get("relations", [])
                return entities, relations
        except Exception as e:
            print(f"LLM extraction error: {e}")

        return [], []

    def _store_entities_and_relations(
        self,
        graph_id: str,
        episode_uuid: str,
        text: str,
        entities: List[Dict],
        relations: List[Dict]
    ):
        """将实体和关系存储到 Neo4j"""
        with self.driver.session(database=self.database) as session:
            # 清理关系类型名称中的非法字符
            def clean_rel_type(rel_type: str) -> str:
                return rel_type.replace(" ", "_").replace("-", "_").replace(":", "_")

            # 创建 Episode 节点
            session.run(
                """
                MATCH (p:Project {graph_id: $graph_id})
                CREATE (e:Episode {
                    uuid: $episode_uuid,
                    text: $text,
                    graph_id: $graph_id,
                    created_at: datetime()
                })
                CREATE (p)-[:HAS_EPISODE]->(e)
                """,
                graph_id=graph_id,
                episode_uuid=episode_uuid,
                text=text
            )

            # 第一步：创建所有实体节点
            for entity in entities:
                entity_name = entity.get("name", "").strip()
                entity_type = entity.get("type", "Entity")
                if not entity_name:
                    continue

                session.run(
                    """
                    MERGE (n:Entity {name: $name, graph_id: $graph_id})
                    SET n.type = $type,
                        n.description = $description
                    """,
                    graph_id=graph_id,
                    name=entity_name,
                    type=entity_type,
                    description=entity.get("description", "")
                )

            # 第二步：为每个实体创建与Episode的MENTIONED_IN关系
            for entity in entities:
                entity_name = entity.get("name", "").strip()
                if not entity_name:
                    continue

                session.run(
                    """
                    MATCH (e:Episode {uuid: $episode_uuid, graph_id: $graph_id})
                    MATCH (n:Entity {name: $name, graph_id: $graph_id})
                    MERGE (n)-[:MENTIONED_IN]->(e)
                    """,
                    graph_id=graph_id,
                    episode_uuid=episode_uuid,
                    name=entity_name
                )

            # 第三步：创建实体间的关系（使用简化的 MERGE）
            for relation in relations:
                source = relation.get("source", "").strip()
                target = relation.get("target", "").strip()
                rel_type = clean_rel_type(relation.get("type", "RELATED_TO"))
                description = relation.get("description", "")

                if not source or not target:
                    continue

                # 使用 MERGE 创建关系，然后 SET 属性
                session.run(
                    f"""
                    MATCH (s:Entity {{name: $source, graph_id: $graph_id}})
                    MATCH (t:Entity {{name: $target, graph_id: $graph_id}})
                    MERGE (s)-[r:`{rel_type}`]->(t)
                    SET r.graph_id = $graph_id,
                        r.description = $description
                    """,
                    graph_id=graph_id,
                    source=source,
                    target=target,
                    description=description
                )

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None
    ):
        """
        等待 episodes 处理完成。

        Neo4j 是同步的，数据已经直接存储，所以这个方法是空操作。
        保留这个方法是为了与 Zep API 兼容。
        """
        if progress_callback:
            progress_callback("All data stored in Neo4j (synchronous)", 1.0)
        pass

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """异步构建图谱"""
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Starting graph build...",
            )

            # Step 1: 创建图谱
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"Graph created: {graph_id}",
            )

            # Step 2: 设置本体
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="Ontology set",
            )

            # Step 3: 文本分块
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"Text split into {total_chunks} chunks",
            )

            # Step 4: 处理文本批次
            def progress_cb(msg, prog):
                self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.6),
                    message=msg,
                )

            self.add_text_batches(
                graph_id,
                chunks,
                ontology,
                batch_size,
                progress_callback=progress_cb,
            )

            # Step 5: 获取图谱信息
            self.task_manager.update_task(
                task_id,
                progress=90,
                message="Retrieving graph information...",
            )

            graph_info = self._get_graph_info(graph_id)

            # 完成
            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """获取图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            # 统计实体数量
            node_result = session.run(
                """
                MATCH (n:Entity {graph_id: $graph_id})
                RETURN count(n) AS node_count,
                       collect(distinct n.type) AS types
                """,
                graph_id=graph_id
            )
            node_record = node_result.single()
            node_count = node_record["node_count"] if node_record else 0
            entity_types = [t for t in node_record["types"] if t] if node_record else []

            # 统计关系数量
            edge_result = session.run(
                """
                MATCH ()-[r]->()
                WHERE r.graph_id = $graph_id
                RETURN count(r) AS edge_count
                """,
                graph_id=graph_id
            )
            edge_record = edge_result.single()
            edge_count = edge_record["edge_count"] if edge_record else 0

            return GraphInfo(
                graph_id=graph_id,
                node_count=node_count,
                edge_count=edge_count,
                entity_types=entity_types,
            )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """获取完整图谱数据"""
        with self.driver.session(database=self.database) as session:
            # 获取所有实体节点
            nodes_result = session.run(
                """
                MATCH (n:Entity {graph_id: $graph_id})
                RETURN n.uuid AS uuid,
                       n.name AS name,
                       n.type AS type,
                       n.description AS description,
                       n.created_at AS created_at
                """,
                graph_id=graph_id
            )

            nodes_data = []
            node_map = {}  # name -> uuid
            for record in nodes_result:
                name = record["name"]
                node_uuid = str(uuid.uuid4())  # 生成UUID
                node_map[name] = node_uuid

                nodes_data.append({
                    "uuid": node_uuid,
                    "name": name,
                    "labels": [record["type"]] if record["type"] else ["Entity"],
                    "summary": record.get("description", ""),
                    "attributes": {},
                    "created_at": str(record["created_at"]) if record["created_at"] else None,
                })

            # 获取所有关系
            edges_result = session.run(
                """
                MATCH (s)-[r]->(t)
                WHERE r.graph_id = $graph_id
                RETURN type(r) AS rel_type,
                       s.name AS source_name,
                       t.name AS target_name,
                       r.description AS description,
                       r.created_at AS created_at
                """,
                graph_id=graph_id
            )

            edges_data = []
            for record in edges_result:
                source_name = record["source_name"]
                target_name = record["target_name"]

                if source_name in node_map and target_name in node_map:
                    edges_data.append({
                        "uuid": str(uuid.uuid4()),
                        "name": record["rel_type"].replace("_", " "),
                        "fact": record["description"] or "",
                        "fact_type": record["rel_type"],
                        "source_node_uuid": node_map[source_name],
                        "target_node_uuid": node_map[target_name],
                        "source_node_name": source_name,
                        "target_node_name": target_name,
                        "attributes": {},
                        "created_at": str(record["created_at"]) if record["created_at"] else None,
                        "episodes": [],
                    })

            return {
                "graph_id": graph_id,
                "nodes": nodes_data,
                "edges": edges_data,
                "node_count": len(nodes_data),
                "edge_count": len(edges_data),
            }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
