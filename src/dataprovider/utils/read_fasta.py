
def read_fasta(filename):

    with open(filename, "r") as file:
        file_parts = file.read().split(">")
        file_parts.pop(0)

        return file_parts

