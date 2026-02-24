from graph_rag import run_graph_rag

query = "The petitioner alleges violation of privacy due to arbitrary state action."

result = run_graph_rag(query)

print("Facts:", result["facts"])
print("Sections:", result["sections"])
print("Acts:", result["acts"])
