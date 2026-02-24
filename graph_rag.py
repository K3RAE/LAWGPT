from legal_graph import build_legal_graph
from fact_mapper import extract_facts

def run_graph_rag(case_text):
    G = build_legal_graph()
    facts = extract_facts(case_text)

    sections = set()
    acts = set()

    for fact in facts:
        if fact in G:
            for section in G.successors(fact):
                sections.add(section)
                for act in G.successors(section):
                    acts.add(act)

    return {
        "facts": facts,
        "sections": list(sections),
        "acts": list(acts)
    }
