import yaml

def config_loader(filepath):
    data = None
    with open(filepath, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        yamlfile.close()
    return data

def config_saver(data, filepath):
    with open(filepath, 'w') as yamlfile:
        data1 = yaml.dump(data, yamlfile)
        yamlfile.close()
