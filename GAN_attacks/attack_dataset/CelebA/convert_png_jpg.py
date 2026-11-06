# Load .txt file, and change all .png text to .jpg

def convert_txt_png_to_jpg(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Replace .png with .jpg
            new_line = line.replace('.png', '.jpg')
            outfile.write(new_line)
    print(f"Converted {input_file} to {output_file} by changing .png to .jpg")


# Example usage
convert_txt_png_to_jpg('trainset.txt', 'trainset_jpg.txt')
convert_txt_png_to_jpg('testset.txt', 'testset_jpg.txt')
convert_txt_png_to_jpg('ganset.txt', 'ganset_jpg.txt')
convert_txt_png_to_jpg('ganset_id_sorted.txt', 'ganset_id_sorted_jpg.txt')