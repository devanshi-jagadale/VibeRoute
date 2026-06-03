import json
import os

# 1. Load Graph Data
graph_path = "data/mood_graph.json"
if not os.path.exists(graph_path):
    raise FileNotFoundError(f"Could not find graph data at {graph_path}.")

with open(graph_path, "r") as f:
    graph_data = json.load(f)

id_to_label = {}
nodes_js = []

print("Processing nodes...")
# We support both list of dicts and list of raw strings seamlessly
for index, node in enumerate(graph_data.get("nodes", [])):
    if isinstance(node, dict):
        c_id = node.get("id", index)
        label = node.get("label", f"Cluster {c_id}")
        size = node.get("size", "N/A")
    else:
        # If it's a raw string label, treat it directly!
        c_id = index
        label = str(node)
        size = "Dynamic"
    
    # Map index integers, index strings, and text labels to this node
    id_to_label[str(c_id)] = label
    id_to_label[int(c_id)] = label
    id_to_label[label] = label

    # Base node size visualization
    radius = 25
    
    nodes_js.append({
        "id": label,
        "label": f"{label}" if size == "Dynamic" else f"{label}\n({size} songs)",
        "size": radius,
        "color": "#824519",  # Russet Brown
        "shape": "dot",
        "font": {"color": "#ffffff", "size": 14}
    })

# 3. Process Edges safely matching indices or names
print("Weaving connections...")
edges_js = []
for edge in graph_data.get("edges", []):
    if not isinstance(edge, dict):
        continue
        
    src_raw = edge.get("source")
    tgt_raw = edge.get("target")
    weight = float(edge.get("weight", 0.5))
    
    # Check our dictionary keys across all possible string/integer types
    src_resolved = id_to_label.get(src_raw) or id_to_label.get(str(src_raw))
    tgt_resolved = id_to_label.get(tgt_raw) or id_to_label.get(str(tgt_raw))
    
    if src_resolved and tgt_resolved:
        edges_js.append({
            "from": src_resolved,
            "to": tgt_resolved,
            "value": weight * 7,
            "title": f"Similarity: {weight:.4f}",
            "color": {"color": "#193d2b", "highlight": "#824519"},  # Forest Green line
            "width": max(2, int(weight * 5))
        })

# 4. Generate Inline CDN vis-network HTML Canvas
html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Mood-Aware Playlist Sequencer: Macro Layer</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; background-color: #1a1a1a; font-family: sans-serif; overflow: hidden; }}
        #network-canvas {{ width: 100%; height: 100vh; }}
        .title-overlay {{ position: absolute; top: 15px; left: 20px; color: #ffffff; z-index: 10; pointer-events: none; }}
        .title-overlay h1 {{ margin: 0 0 5px 0; font-size: 20px; letter-spacing: 1px; color: #e6e6e6; }}
        .title-overlay p {{ margin: 0; font-size: 12px; color: #888888; }}
    </style>
</head>
<body>
    <div class="title-overlay">
        <h1>Mood Cluster Connectivity Graph</h1>
        <p>Nodes: {len(nodes_js)} robust segments | Edges: {len(edges_js)} verified boundaries</p>
    </div>
    <div id="network-canvas"></div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes_js)});
        var edges = new vis.DataSet({json.dumps(edges_js)});

        var container = document.getElementById('network-canvas');
        var data = {{ nodes: nodes, edges: edges }};
        
        var options = {{
            nodes: {{ borderWidth: 2, borderColor: '#ffffff' }},
            edges: {{ smooth: {{ type: 'continuous' }} }},
            physics: {{
                barnesHut: {{
                    gravitationalConstant: -25000,
                    centralGravity: 0.3,
                    springLength: 170,
                    springConstant: 0.04
                }},
                minVelocity: 0.75
            }}
        }};
        
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""

# 5. Write pure output file
output_html = "mood_graph_vis.html"
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"\n✅ Success! Webpage created without errors.")
print(f"👉 Generated canvas file: {os.path.abspath(output_html)}")