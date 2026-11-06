def match_ganset_identities(ganset_file, identities_file, output_file, sorted_output_file, sorted_id_output_file):
    # Read ganset.txt
    with open(ganset_file, 'r') as f:
        ganset_filenames = set(line.strip() for line in f)
    
    # Read identities.txt and create a mapping
    identity_map = {}
    with open(identities_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                jpg_filename, identity = parts
                png_filename = jpg_filename.replace('.jpg', '.png')
                identity_map[png_filename] = identity
    
    # Match PNG filenames with identities
    output_lines = []
    unique_identities = set()
    for filename in ganset_filenames:
        identity = identity_map.get(filename, "Not Found")
        output_lines.append((filename, identity))
        if identity != "Not Found":
            unique_identities.add(identity)
    
    # Write to output file in txt format
    with open(output_file, 'w') as f:
        for filename, identity in output_lines:
            f.write(f"{filename} {identity}\n")
    
    print(f"Output saved to {output_file}")
    
    # Sort by identity and write to sorted output file, sorted by filename
    sorted_lines = sorted(output_lines, key=lambda x: x[0])
    with open(sorted_output_file, 'w') as f:
        for filename, identity in sorted_lines:
            f.write(f"{filename} {identity}\n")

    # Sort by identity and write to sorted output file, sorted by id
    sorted_lines = sorted(output_lines, key=lambda x: x[1])
    with open(sorted_id_output_file, 'w') as f:
        for filename, identity in sorted_lines:
            f.write(f"{filename} {identity}\n")
            
    
    print(f"Sorted output saved to {sorted_output_file}")
    print(f"Total unique identities: {len(unique_identities)}")

# Example usage
match_ganset_identities('ganset.txt', 'identity_CelebA.txt', 'ganset_id.txt', 'ganset_id_sorted.txt', 'ganset_id_sorted_by_id.txt')
