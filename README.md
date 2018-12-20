# Rain sensing windshield wipers
Deep learning network to activate the windshield wipers based on the amount of rain on the windshield

#### Clone the project and install dependencies

Install [pipenv](https://github.com/pypa/pipenv)
```bash
$ brew install pipenv
```

Install Python packages
```bash
$ pipenv install --dev
```

#### Run the project locally

Open pipenv shell
```bash
$ pipenv shell
```

Generate the `tfrecord` files (run in the shell)
```bash
$ python create_dataset.py
```

Train the model (run in the shell)
```bash
$ python train.py
```
