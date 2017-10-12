

def write_to_fasta(entries, filename):

    with open(filename, "w") as file:
        for lines in entries:
            file.write(">")
            for line in lines:
                file.write("%s\n" % line)
            file.write("\n")