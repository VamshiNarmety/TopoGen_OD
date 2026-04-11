"""
Centrality Analysis Module
Computes multiple centrality measures on the transport network to identify 
critical nodes for disruption resilience.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
from tqdm import tqdm


def load_edges(edges_parquet_path):
    """Load edge list from parquet file."""
    df = pd.read_parquet(edges_parquet_path)
    print(f"Loaded {len(df)} edges from {edges_parquet_path}")
    return df


def build_graph(edges_df):
    """Build directed weighted graph from edge list."""
    G = nx.DiGraph()
    
    # Determine column names (support multiple naming conventions)
    if 'origin' in edges_df.columns and 'destination' in edges_df.columns:
        source_col, target_col = 'origin', 'destination'
    elif 'source' in edges_df.columns and 'target' in edges_df.columns:
        source_col, target_col = 'source', 'target'
    else:
        source_col, target_col = edges_df.columns[0], edges_df.columns[1]
    
    # Determine weight column
    weight_col = None
    for col in ['trip_count', 'weight', 'trips']:
        if col in edges_df.columns:
            weight_col = col
            break
    if weight_col is None:
        weight_col = edges_df.columns[-1]
    
    for _, row in edges_df.iterrows():
        source = row[source_col]
        target = row[target_col]
        weight = row[weight_col]
        
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight)
    
    print(f"Built directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_centrality_measures(G_directed, G_undirected):
    """
    Compute multiple centrality measures on both directed and undirected graphs.
    Returns a dictionary of centrality measures.
    """
    centrality_dict = {}
    
    print("Computing centrality measures...")
    
    # Directed graph centrality measures
    print("  - In-degree centrality (directed)...")
    in_degree_cent = nx.in_degree_centrality(G_directed)
    centrality_dict['in_degree_centrality'] = in_degree_cent
    
    print("  - Out-degree centrality (directed)...")
    out_degree_cent = nx.out_degree_centrality(G_directed)
    centrality_dict['out_degree_centrality'] = out_degree_cent
    
    # Undirected graph centrality measures (often more interpretable for transport)
    print("  - Betweenness centrality (undirected)...")
    betweenness = nx.betweenness_centrality(G_undirected, weight='weight')
    centrality_dict['betweenness_centrality'] = betweenness
    
    print("  - Closeness centrality (undirected)...")
    closeness = nx.closeness_centrality(G_undirected, distance='weight')
    centrality_dict['closeness_centrality'] = closeness
    
    print("  - Eigenvector centrality (undirected)...")
    try:
        eigenvector = nx.eigenvector_centrality(
            G_undirected, weight='weight', max_iter=1000
        )
        centrality_dict['eigenvector_centrality'] = eigenvector
    except nx.NetworkXError:
        print("    Warning: Eigenvector centrality failed (possibly disconnected). Using 0 fallback.")
        eigenvector = {node: 0 for node in G_undirected.nodes()}
        centrality_dict['eigenvector_centrality'] = eigenvector
    
    print("  - PageRank (on directed graph)...")
    pagerank = nx.pagerank(G_directed, weight='weight')
    centrality_dict['pagerank'] = pagerank
    
    return centrality_dict


def centrality_table(nodes, centrality_dict):
    """Convert centrality measures to DataFrame."""
    data = {'node_id': nodes}
    
    for measure_name, measure_values in centrality_dict.items():
        data[measure_name] = [measure_values.get(node, 0.0) for node in nodes]
    
    df = pd.DataFrame(data)
    return df


def top_k(centrality_dict, k=10):
    """Extract top-k nodes for each centrality measure."""
    top_dict = {}
    
    for measure_name, measure_values in centrality_dict.items():
        sorted_nodes = sorted(
            measure_values.items(), key=lambda x: x[1], reverse=True
        )
        top_dict[measure_name] = [
            {'rank': i+1, 'node_id': node, 'value': value}
            for i, (node, value) in enumerate(sorted_nodes[:k])
        ]
    
    return top_dict


def summarize_centrality(centrality_df, centrality_dict, top_k_dict):
    """Generate summary statistics for centrality measures."""
    summary = {
        'total_nodes': int(len(centrality_df)),
        'measures': {}
    }
    
    for measure_name in centrality_dict.keys():
        col = centrality_df[measure_name]
        summary['measures'][measure_name] = {
            'mean': float(col.mean()),
            'std': float(col.std()),
            'min': float(col.min()),
            'max': float(col.max()),
            'top_10': [
                {'rank': int(item['rank']), 'node_id': int(item['node_id']), 'value': float(item['value'])}
                for item in top_k_dict[measure_name]
            ]
        }
    
    return summary


def main(args):
    """Main execution: load graph, compute centrality, save outputs."""
    
    # Load edges
    edges_df = load_edges(args.input)
    
    # Build directed graph
    G_directed = build_graph(edges_df)
    
    # Build undirected graph for symmetric centrality measures
    G_undirected = G_directed.to_undirected()
    print(f"Created undirected projection: {G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges")
    
    # Compute all centrality measures
    centrality_dict = compute_centrality_measures(G_directed, G_undirected)
    
    # Create centrality DataFrame
    nodes = sorted(list(G_directed.nodes()))
    centrality_df = centrality_table(nodes, centrality_dict)
    
    # Get top-10 for each measure
    top_k_dict = top_k(centrality_dict, k=10)
    
    # Summarize
    summary = summarize_centrality(centrality_df, centrality_dict, top_k_dict)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save centrality DataFrame
    centrality_path = output_dir / f"{args.prefix}_centrality.parquet"
    centrality_df.to_parquet(centrality_path, index=False)
    print(f"\nSaved centrality table to {centrality_path}")
    
    # Save summary JSON
    summary_path = output_dir / f"{args.prefix}_centrality_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved centrality summary to {summary_path}")
    
    print(f"\n✓ Centrality analysis complete!")
    print(f"  Nodes: {summary['total_nodes']}")
    print(f"  Measures: {list(summary['measures'].keys())}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute centrality measures on transport network')
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to edges parquet file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/network',
        help='Output directory for centrality files'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='centrality',
        help='Prefix for output files'
    )
    
    args = parser.parse_args()
    main(args)
