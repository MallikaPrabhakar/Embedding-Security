# Load trainset.txt, testset.txt, ganset.txt, extract all image filenames, concatenate, and output

def get_all_filenames(trainset_file, testset_file, ganset_file, output_file):
    with open(ganset_file, 'r') as f:
        ganset_filenames = set(line.strip() for line in f)

    with open(testset_file, 'r') as f:
        testset_filenames = set(line.strip().split()[0] for line in f)

    with open(trainset_file, 'r') as f:
        trainset_filenames = set(line.strip().split()[0] for line in f)

    with open(output_file, 'w') as f:
        for filename in trainset_filenames:
            f.write(f"{filename}\n")
        for filename in testset_filenames:
            f.write(f"{filename}\n")
        for filename in ganset_filenames:
            f.write(f"{filename}\n")

get_all_filenames("trainset_jpg.txt", "testset_jpg.txt", "ganset_jpg.txt", "all_filenames_jpg.txt")
    

