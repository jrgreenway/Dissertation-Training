# Training and Validation

This project is a part of the overall Dissertation project for James Greenway.

## Instructions

The recommended way to use this repository is by building the docker image and attaching this file as a volume:

Run `docker build [NAME]` with the project folder as the working directory, which will build the docker image with `dockerfile`. After this to run use `docker run -it --rm -v "$(pwd)":/app NAME`. If there is a device with GPU support, you can specify this by adding `--gpus all` or `--gpus device=NUM` for devices with one GPU or to specify which GPU to use respectively.

Alternatively, a virtual environment may be set up in the command line with `python -m venv .venv`, activating it with `.venv\Scripts\activate` and then doing `pip install -r requirements.txt`, athough this is not guarenteed to work on every device.

### Data

To generate data, run `python data_gen.py` in the command line, with parameters for how the data is generated in `GLOBAL_VAR.py`. This generates data in the form of JSON in `results/events` folder.

### Training

To run the training script run `training.py --model [MODEL] --fraction [DATA_SIZE]`, where DATA*SIZE is a number between 0 and 1 represting the portion of the dataset to train on and MODEL is the name of the model to be trained. Alternatively if in the docker image, any of the `run*[SIZE].sh` scripts will run all the different models with the apropriate volume of data. At the end of the script, the metrics from the model are saved in results, as well as the model for each fold and the tokeniser for each fold.

### Other Scripts

The `benchmark.py` script benchmarks the models in `models/` on the device, putting the results in benchmark.json in seconds.
`cuda_check.py` is a useful script to determine whether there is a GPU avalible for the environment to use.
