import numpy as np
import matplotlib.pyplot as plt
import os
from deepface import DeepFace
import seaborn as sns
import argparse
import time
import cv2


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
        
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def reconstruction_accuracy(original_images, reconstructed_images, metric='euclidean_l2', model='Facenet512', detector_backend='skip', generate_plot=False, visualize=False, 
                            distances_path="", plot_path=""):
    distances = []

    # metric is normalised euclidean distance: computed for 2000 pairs of images (1000 same and different images of celeba dataset)
    if model == 'Facenet512':
        threshold = 1.1027
    else:
        threshold = 1.1007

    print(f"Using {metric} metric with threshold {threshold} for {model} model with {detector_backend} detector backend.")
    for i in range(len(reconstructed_images)):
        if visualize:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(original_images[i])
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_images[i])
            plt.title('Reconstructed Image')
            plt.axis('off')
            plt.show()

        res = DeepFace.verify(original_images[i], reconstructed_images[i], model_name=model, distance_metric=metric, detector_backend=detector_backend, normalization='ArcFace', enforce_detection=False, threshold=threshold)

        distances.append(res['distance'])
    
    distances = np.array(distances)
    accuracy = np.mean(distances < threshold)
    print('Accuracy: ', accuracy, "Threshold: ", threshold)

    if generate_plot:
        np.save(distances_path, distances)
        sns.kdeplot(distances)
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.xlabel('{} Distance'.format(metric))
        plt.ylabel('Density')
        plt.savefig(plot_path)
        #plt.show()
        print(f"Saved distances and plot to {distances_path} and {plot_path}")
        

def reconstruction_accuracy_from_distances(distances_path):
    distances = np.load(distances_path)
    threshold = 1.1027
    accuracy = np.mean(distances < threshold)
    print('Accuracy: ', accuracy)
        

def main():
    args = create_argparser().parse_args()
    eval_dset = args.eval_dset
    output_dir = args.output_dir
    original_dir = args.original_dir
    detector_backend = args.detector_backend
    prot_scheme = args.prot_scheme
    same_r = args.same_r
    samples = args.samples
    metric = args.metric
    model = args.model
    # os.environ["DEVICE_ID"] = args.device_id

    original_path = os.path.join(os.getcwd(), original_dir, f'{prot_scheme}_{same_r}', eval_dset, "images")
    reconstructed_path = os.path.join(os.getcwd(), original_dir, f'{prot_scheme}_{same_r}', eval_dset, "reconstructed_images")

    result_path = os.path.join(os.getcwd(), output_dir, f'{prot_scheme}_{same_r}', eval_dset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    distances_path = os.path.join(result_path, "distances")
    plot_path = os.path.join(result_path, "plot.png")

    print("Evaluating", output_dir, f'{prot_scheme}_{same_r}', eval_dset)
    start_time = time.time()

    # Load reconstructed images to be evaluated
    reconstructed_images = [] 
    original_images = []

    image_paths = os.listdir(reconstructed_path)
    image_paths.sort()
    print("Reconstructed images: ", len(image_paths))
    # Load same number of original images
    for i in range(samples):
        og_img = np.load(os.path.join(original_path, image_paths[i]))
        re_img = np.load(os.path.join(reconstructed_path, image_paths[i]))
        og_img = np.moveaxis(og_img, 0, -1)
        re_img = np.moveaxis(re_img, 0, -1)
        # og_img = og_img * 255.0
        # re_img = re_img * 255.0
        og_img = og_img.astype(np.uint8)
        re_img = re_img.astype(np.uint8)
        og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2RGB)
        re_img = cv2.cvtColor(re_img, cv2.COLOR_BGR2RGB)
        original_images.append(og_img)
        reconstructed_images.append(re_img)
        # print(og_img.flatten()[:5], re_img.flatten()[:5])
        # print("Original image: ", og_img[0, 0])
        # print("Reconstructed image: ", re_img[0, 0])

        # emb1 = DeepFace.represent(og_img, model_name=model, detector_backend='skip', enforce_detection=False)
        # emb2 = np.load(os.path.join(original_path.replace("images", "embeddings"), image_paths[i]))
        # print(np.linalg.norm(emb1[0]['embedding'] - emb2.flatten()))

    reconstruction_accuracy(original_images, reconstructed_images, metric, model, detector_backend, generate_plot=True, visualize=False, distances_path=distances_path, plot_path=plot_path)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed for evaluation {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )

def create_argparser():
    defaults = dict(
        device_id="0",
        eval_dset="test",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--detector_backend", type=str, default='skip')
    parser.add_argument("--prot_scheme", type=str, required=True)
    parser.add_argument("--same_r", type=str, required=True)
    parser.add_argument("--samples", type=int, required=True)

    return parser

if __name__ == "__main__":
    main()