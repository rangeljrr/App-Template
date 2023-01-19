import pickle as pkl

def serialize_model(object, path):
    """ This function will take an object and dump it as a 
        pkl object to the specified path """
        
    with open(path, "wb") as output_file:
        pkl.dump(object, output_file)