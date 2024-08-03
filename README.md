# Towards Learning Image Similarity from General Triplet Labels

Are fine-grained classes the only practical way to express detailed visual similarity? The <a href="https://openaccess.thecvf.com/content/CVPR2024W/FGVC11/papers/Dondera_Towards_Learning_Image_Similarity_from_General_Triplet_Labels_CVPRW_2024_paper.pdf">"Towards Learning Image Similarity from General Triplet Labels" paper</a> investigates a triplet based definition of similarity and addresses its challenges with a knowledge transfer approach.

## Installation

### Code

We suggest to use a conda environment and to install the necessary packages with

```
pip install -r requirements.txt
```

The code has been tested on a RTX 3090 GPU with CUDA 11.8.

### Data

Please follow the instructions in the <a href="https://github.com/greenfieldvision/ditdml">ditdml repository</a> to install the THINGS, IHSJ and Yummly datasets. Use /data/metric_learning/datasets/things_raw, ihsj_raw, yummly_raw as root folders, or alternatively modify the base paths in the yaml configuration files.

The CUB, CARS and SOP datasets are easy to find (<a href="https://github.com/htdt/hyp_metric?tab=readme-ov-file#datasets">some links</a>) and they should also be downloaded to /data/metric_learning/datasets/cub_raw, cars_raw and sop_raw or the paths should be modified in the configuration files.

## Training and Evaluation

Commands for all the runs in the paper are in `commands.txt` and they must be executed from the parent folder of tdml. Representative examples are below.

```
python tdml/train.py configuration_file_name=tdml/configurations/things/things_p_ste.yaml num_repetitions=5

python tdml/save_embeddings.py configuration_file_name=tdml/configurations/things/things_p_ste.yaml model_uris=Psycho,embedding_size:128,beta:0.0/modeling/runs/run-0111..0115/checkpoint-best.pth output_directory_name=modeling/intermediate/datasets/things_se new_source_embeddings_id=pt_ste5

python tdml/train.py configuration_file_name=tdml/configurations/things/things_se_rkd.yaml

python tdml/evaluate.py configuration_file_name=tdml/configurations/things/things_e.yaml model_uri=ResNet50,embedding_size:128,frozen_variables:bn,num_nonlinearity_layers:0/modeling/runs/run-0121/checkpoint-best.pth evaluation_type=triplets


python tdml/train.py configuration_file_name=tdml/configurations/cars/cars_se_stmr_ns_vits.yaml

python tdml/evaluate.py configuration_file_name=tdml/configurations/cars/cars_e.yaml model_uri=ViTS,embedding_size:384,num_nonlinearity_layers:0/modeling/runs/run-0027/checkpoint-best.pth evaluation_type=classes
```
