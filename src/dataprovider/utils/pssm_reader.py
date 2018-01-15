
def read_pssm_file(name):

    path = "datasets/tmseg/data/pssm/uniref90j5/"
    filename = path + name + ".pssm"

    rows = []
    with open(filename, "r") as file:
        lines = file.read().split("\n")

        for line in lines:
            line_parts = line.split()
            try:
                if len(line_parts) > 0:
                    int(line_parts[0])
                    rows.append(list(map(int, line_parts[2:22])))
            except ValueError:
                continue

    return rows
