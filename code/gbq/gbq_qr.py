from utils import parse_args
from run import run_gbq_flu_model

def main():
    # parse arguments
    model_config, run_config = parse_args()
    
    # fit model and generate predictions
    run_gbq_flu_model(model_config, run_config)


if __name__ == '__main__':
    main()
