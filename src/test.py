
def read_fasta(filename):

    with open(filename, "r") as file:
        file_parts = file.read().split(">")
        file_parts.pop(0)

        return file_parts


path = "datasets/tmseg/data/sets/"
fasta_path = "unmasked_hval0/"

opm_set1 = "opm_set1"
opm_set2 = "opm_set2"
opm_set3 = "opm_set3"
opm_set4 = "opm_set4"

sets = [opm_set1, opm_set2, opm_set3, opm_set4]

for set in sets:
    parts = read_fasta(path + fasta_path + set + ".fasta")
    print(len(parts))

