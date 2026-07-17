import networkx as nx
from typing import List
from src.config import ExtractedKnowledge

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_knowledge(self, knowledge_list: List[ExtractedKnowledge]):
        """
        Adds deduplicated entities and relationships to the NetworkX graph.
        """
        for k in knowledge_list:
            for ent in k.entities:
                if not self.graph.has_node(ent.name):
                    self.graph.add_node(ent.name, type=ent.type, description=ent.description, source_chunk_id=getattr(ent, "source_chunk_id", None))
                else:
                    # Update description if it's currently None
                    if not self.graph.nodes[ent.name].get('description') and ent.description:
                        self.graph.nodes[ent.name]['description'] = ent.description
            
            for rel in k.relationships:
                # Add edge, if it exists, maybe increment weight or add source
                if self.graph.has_edge(rel.source_entity, rel.target_entity):
                    # Edge exists, maybe append description or chunk id
                    pass
                else:
                    self.graph.add_edge(
                        rel.source_entity, 
                        rel.target_entity, 
                        relation=rel.relation_type,
                        description=rel.description,
                        source_chunk_id=getattr(rel, "source_chunk_id", None)
                    )

    def save_graph(self, filepath: str):
        """
        Saves the graph to GraphML format.
        """
        nx.write_graphml(self.graph, filepath)
        print(f"Graph saved to {filepath} with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges.")

    def load_graph(self, filepath: str):
        """
        Loads the graph from GraphML format.
        """
        self.graph = nx.read_graphml(filepath)
        print(f"Graph loaded from {filepath}.")
