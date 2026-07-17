from pyvis.network import Network
import networkx as nx

class GraphVisualizer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        
        # Map node types to colors
        self.color_map = {
            "Paper": "#E57373",      # Red
            "Author": "#BA68C8",     # Purple
            "Institution": "#7986CB",# Indigo
            "Method": "#81C784",     # Green
            "Dataset": "#FFB74D",    # Orange
            "Metric": "#4DD0E1",     # Cyan
            "Finding": "#FFF176",    # Yellow
            "Limitation": "#A1887F", # Brown
            "Task": "#F06292"        # Pink
        }

    def generate_html(self, output_path: str = "data/processed/graph_visualization.html"):
        """
        Generates an interactive HTML visualization using Pyvis.
        """
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
        
        # Add nodes with colors based on type
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            color = self.color_map.get(node_type, "#9E9E9E") # Default grey
            
            title = f"Type: {node_type}"
            if 'description' in data and data['description']:
                title += f"\nDesc: {data['description']}"
                
            net.add_node(node, label=node, title=title, color=color, size=15)
            
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            relation = data.get('relation', '')
            net.add_edge(source, target, title=relation, label=relation)
            
        # Physics config for better layout
        net.repulsion(node_distance=200, spring_length=200)
        
        net.write_html(output_path)
        print(f"Visualization generated at {output_path}")
