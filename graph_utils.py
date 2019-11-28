import dgl

def build_graph(graph_df):
    g = dgl.DGLGraph()
    g.add_nodes(graph_df.shape[0])
    pairs_edges = set()
    for _, row in graph_df.iterrows():
        node_i = row["node_id"]
        to_nodes = row["to_nodes"]
        pairs_edges.update(list(zip(len(to_nodes) * [node_i], to_nodes)))
    # add reversed pairs of nodes
    src, dst = list(zip(*(list(pairs_edges))))
    pairs_edges.update(list(zip(dst, src)))
    # add edges to graph: 
    # between nodes a and b there are edges a->b and a<-b if they are connected at least once in data
    src, dst = list(zip(*(list(pairs_edges))))
    g.add_edges(src, dst)
    return g
