import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from deepface import DeepFace
import seaborn as sns
import argparse
import time

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
    if model == 'Facenet512':
        threshold = 1.1027
    else:
        threshold = 1.1007

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
    print('Accuracy: ', accuracy)

    if generate_plot:
        np.save(distances_path, distances)
        sns.kdeplot(distances)
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.xlabel('{} Distance'.format(metric))
        plt.ylabel('Density')
        plt.savefig(plot_path)
        #plt.show()
        print(f"Saved distances and plot to {distances_path} and {plot_path}")
        

def main():
    args = create_argparser().parse_args()
    eval_dset = args.eval_dset
    output_dir = args.output_dir
    original_dir = args.original_dir
    metric = args.metric
    model = args.model
    detector_backend = args.detector_backend
    os.environ["DEVICE_ID"] = args.device_id

    if args.result_file:
        result_path = os.path.join(output_dir, args.result_file)
    else:
        result_path = os.path.join(output_dir, f"{eval_dset}_samples_1000x64x64x3.npz")
    original_path = os.path.join(original_dir, eval_dset, "images")
    distances_path = os.path.join(output_dir, f"{eval_dset}_distances")
    plot_path = os.path.join(output_dir, f"{eval_dset}_plot.png")

    print("Evaluating", output_dir, eval_dset)
    start_time = time.time()

    # Load reconstructed images to be evaluated
    result = np.load(result_path)
    reconstructed_images = result['arr_0'] # reconstructed images obtained by sampling from model
    original_images = [] # read images for original path

    image_paths = os.listdir(original_path)
    image_paths.sort()
    # Load same number of original images
    for i in range(len(reconstructed_images)):
        img = cv2.imread(os.path.join(original_path, image_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_images.append(img)

    reconstruction_accuracy(original_images, reconstructed_images, metric, model, detector_backend, generate_plot=True, visualize=False, distances_path=distances_path, plot_path=plot_path)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed for evaluation {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )

def create_argparser():
    defaults = dict(
        device_id="0",
        eval_dset="",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--detector_backend", type=str, default='skip')
    parser.add_argument("--result_file", type=str, default=False)
    return parser


if __name__ == "__main__":
    main()