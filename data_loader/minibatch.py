"""Because we want to provide accessbility over temporal graphs, we design a new
store format: 
    edges: (from_node_id, to_node_id, timestamp)
    adjacency list: from_node_id: (to_node_id, timestamp)
"""
