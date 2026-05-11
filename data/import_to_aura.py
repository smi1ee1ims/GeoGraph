"""
导入图数据到 Neo4j Aura
用法: python import_to_aura.py
"""
import json
from neo4j import GraphDatabase

# Aura 连接信息
AURA_URI = "neo4j+s://4bc8c2ea.databases.neo4j.io"
AURA_USER = "4bc8c2ea"
AURA_PASSWORD = "lIO1A7Xg1mdhgfvTSN6Ug9Ev0vm2WQ10XKXgU9XkG_w"
  
def import_data(tx, nodes, edges):
    """导入节点和关系"""

    # 建立 node_id -> string_id 的映射
    node_id_map = {}  # node_id(整数) -> props.id(字符串)

    # 导入节点
    for node in nodes:
        node_id = node.get('node_id')  # 整数 0,1,2...
        label = node.get('label', 'Node')
        props = node.get('props', {})
        string_id = props.get('id', f'node_{node_id}')  # 字符串 id

        node_id_map[node_id] = string_id

        tx.run(f"""
            MERGE (n:{label} {{id: $id}})
            SET n += $props
        """, id=string_id, props=props)

    print(f"导入 {len(nodes)} 个节点")

    # 导入关系
    for edge in edges:
        source_int_id = edge.get('source_id')  # 整数
        target_int_id = edge.get('target_id')  # 整数
        rel_type = edge.get('rel_type', 'RELATES')

        source_id = node_id_map.get(source_int_id)
        target_id = node_id_map.get(target_int_id)

        if source_id and target_id:
            # 使用反引号处理中文关系类型名
            tx.run(f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                MERGE (a)-[r:`{rel_type}`]->(b)
            """, source_id=source_id, target_id=target_id)

    print(f"导入 {len(edges)} 条关系")

def main():
    # 读取数据文件（utf-8-sig 处理 BOM）
    with open('nodes.json', 'r', encoding='utf-8-sig') as f:
        nodes = json.load(f)

    with open('edges.json', 'r', encoding='utf-8-sig') as f:
        edges = json.load(f)

    print(f"读取到 {len(nodes)} 个节点, {len(edges)} 条边")

    # 连接 Aura 并导入
    driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PASSWORD))

    with driver.session() as session:
        session.execute_write(import_data, nodes, edges)

    driver.close()
    print("数据导入完成!")

if __name__ == "__main__":
    main()
