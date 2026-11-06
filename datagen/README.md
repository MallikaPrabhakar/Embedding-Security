To generate the preprocessed images and embeddings from the raw datasets, run the following command:

```bash
python generate_embeddings.py --dataset {dataset} --model {model}
```

Where `{dataset}` can be `celeba, lfw, casia_webface` and the `{model}` can be `Facenet512, ArcFace or VGG-Face`

To generate all preprocessed embeddings used in our paper, you can simply run:

```bash
bash datagen.sh
```