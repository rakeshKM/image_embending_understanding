import yaml
import data

def run(config):
    Dataset = data.Dataset(config['datapath'])
    Dataset.describe()
    print("-----COMPLETE------")

# This is needed to run this file as a script rather than import it as a module
if __name__ == "__main__":

    # Load the configuration file
    with open('config.yaml') as p:
        config = yaml.safe_load(p)
        
    run(config)