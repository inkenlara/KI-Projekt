from utils import get_ordering, get_splits

def main():
    # Create all mandatory documents
    ordering = get_ordering()
    splits = get_splits()

    current_part_id = 0
    current_graph_id = 0
    gwp_a = []
    gwp_graph_indicator = []
    gwp_graph_labels = []
    gwp_node_labels = []
    for x, y in zip(splits["x_train"], splits["y_train"]):
        ordered = ordering.sort(x)
        adjacency_matrix = y.get_adjacency_matrix(ordered)

        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if j < i:
                    continue
                if adjacency_matrix[i][j] == 1:
                    gwp_a.append((current_part_id+j,current_part_id+i)) # Adjacency of i and j
        
            gwp_graph_indicator.append(current_graph_id)
            gwp_node_labels.append((ordered[i].get_family_id(), ordered[i].get_part_id()))
            current_part_id += 1

        gwp_graph_labels.append(1)
        current_graph_id += 1

    with open('GRAN/transformed_data/GWP_A.txt', 'a') as f:
        for con in gwp_a:
            f.write(str(con[0])+","+str(con[1])+"\n")
    with open('GRAN/transformed_data/GWP_graph_indicator.txt', 'a') as f:
        for indicator in gwp_graph_indicator:
            f.write(str(indicator)+"\n")
    with open('GRAN/transformed_data/GWP_graph_labels.txt', 'a') as f:
        for label in gwp_graph_labels:
            f.write(str(label)+"\n")
    with open('GRAN/transformed_data/GWP_node_labels.txt', 'a') as f:
        for labels in gwp_node_labels:
            f.write(str(labels[0])+","+str(labels[1])+"\n")

if __name__ == "__main__":
    main()