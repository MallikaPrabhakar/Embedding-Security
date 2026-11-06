Following are the sources for the datasets we have used:

- CelebA: [Webpage](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) 
- LFW: [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- CASIA-Webface: [Google Drive](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view)

Please manually download, unzip and place the datasets in the `raw/` subdirectory in this folder. After downloading all datasets, ignoring the files not used in this repository, the directory structure should look like this:
```
Embedding-Security/
├── datasets/
│   └── raw/
│       ├── faces_webface_112x112/        # Unzipped files and folders
│       │   └── *
│       ├── lfw/
│       │   ├── lfw-deepfunneled/         # Unzipped files and folders
│       │   │   └── lfw-deepfunneled/
│       │   │       └── {person_name}/
│       │   │           └── *
│       │   └── *.csv                     # Unzipped files and folders
│       └── CelebA/                       # Unzipped files and folders
│           ├── Anno/
│           │   └── *.txt                 
│           ├── Eval/
│           │   └── *.txt                 
│           ├── Img/
│           │   └── img_align_celeba/
│           │       └── *.jpg
│           └── README.txt
```

The preprocessed `pp/` datasets (Facenet512 embeddings + aligned images) are provided by
`download_all.sh`. The directory structure is as follows:

```
Embedding-Security/
├── datasets/
│   ├── raw/                              # Raw / original datasets
│   │   └── *
│   └── pp/                               # Preprocessed datasets
│       ├── casia_webface/
│       │   ├── embeddings_facenet/*
│       │   ├── images/*
│       │   └── img_id_map.csv
│       ├── celeba/
│       │   ├── embeddings_facenet/*
│       │   ├── embeddings_gan_facenet/*
│       │   ├── images/*
│       │   └── img_id_map.csv
│       └── lfw/
│           ├── embeddings_facenet/*
│           ├── images/*
│           └── img_id_map.csv
```
