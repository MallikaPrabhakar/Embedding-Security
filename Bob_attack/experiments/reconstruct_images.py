import numpy as np
import os, sys
import argparse
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'TrainNetwork'))
sys.path.append(os.path.join(os.getcwd(), 'eval'))
# print(sys.path)
from transformers import InversionTransformer
import time


def get_reconstructed_image(emb_model, inv_model, image, device, img_id=None):
    image = torch.from_numpy(image).to(device)
    embedding = emb_model(torch.from_numpy(image).unsqueeze(0).to(device), img_id = img_id)
    embedding = embedding.cpu()
    reconstructed_image = inv_model.transform(embedding.cpu())[0].data/255
    return reconstructed_image


def main():
    args = create_argparser().parse_args()
    original_dir = args.original_dir
    generator_checkpoint = args.generator_checkpoint
    eval_dset = args.eval_dset
    model = args.model
    prot_scheme = args.prot_scheme
    same_r = args.same_r
    embedding_size = args.embedding_size
    samples = args.samples

    original_path = os.path.join(os.getcwd(), original_dir, f'{prot_scheme}_{same_r}', eval_dset, "embeddings")
    reconstructed_path = os.path.join(os.getcwd(), original_dir, f'{prot_scheme}_{same_r}', eval_dset, "reconstructed_images")
    if not os.path.exists(reconstructed_path):
        os.makedirs(reconstructed_path)

    generator_path = f'{os.getcwd()}/TrainNetwork/training_files/{model}/{prot_scheme}_{same_r}/models/Generator_{generator_checkpoint}.pth'
    
    inv_model = InversionTransformer(checkpoint=generator_path, length_of_embedding=embedding_size, model=model)

    print("Sampling", original_dir, model, f'{prot_scheme}_{same_r}',  eval_dset)
    start_time = time.time()

    emb_paths = os.listdir(original_path)
    emb_paths.sort()

    for i in range(samples):
        if i % 100 == 0:
            print(f"Processing {i}/{samples} images")
        embedding = np.load(os.path.join(original_path, emb_paths[i]))
        embedding = np.reshape(embedding, (1, embedding_size))
        embedding = torch.from_numpy(embedding)

        reconstructed_image = inv_model.transform(embedding)
        reconstructed_image = reconstructed_image[0].data
        reconstructed_image = reconstructed_image.astype(np.uint8)
        np.save(os.path.join(reconstructed_path, emb_paths[i]), reconstructed_image)
        

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed for generating reconsrtucted images {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--generator_checkpoint", type=str, required=True)
    parser.add_argument("--eval_dset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=['facenet'])
    parser.add_argument("--prot_scheme", type=str, required=True,
                        help="output-dir tag base; e.g. unprotected, facialfe_e8, facialfe_leech_centered, ourfe")
    parser.add_argument("--same_r", type=str, required=True, choices=['none', 'true', 'false'])
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=1000)
    return parser


if __name__ == "__main__":
    main()