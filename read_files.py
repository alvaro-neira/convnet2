file1 = open('catalog.txt', 'r')
Lines = file1.readlines()

# Strips the newline character
for line in Lines:
    print(f"{line.strip()}")