import networkx as nx

def build_legal_graph():
    G = nx.DiGraph()

    # ---- Act Nodes ----
    G.add_node("Constitution of India", type="Act")
    G.add_node("Indian Penal Code", type="Act")

    # ---- Section Nodes ----
    G.add_node("Article 21", type="Section")
    G.add_node("Article 14", type="Section")
    G.add_node("Section 300 IPC", type="Section")

    # ---- Fact Nodes ----
    G.add_node("Privacy Violation", type="Fact")
    G.add_node("Arbitrary State Action", type="Fact")
    G.add_node("Criminal Liability", type="Fact")

    # ---- Edges ----
    G.add_edge("Privacy Violation", "Article 21", relation="cites")
    G.add_edge("Arbitrary State Action", "Article 14", relation="cites")
    G.add_edge("Criminal Liability", "Section 300 IPC", relation="cites")

    G.add_edge("Article 21", "Constitution of India", relation="part_of")
    G.add_edge("Article 14", "Constitution of India", relation="part_of")
    G.add_edge("Section 300 IPC", "Indian Penal Code", relation="part_of")

    return G
