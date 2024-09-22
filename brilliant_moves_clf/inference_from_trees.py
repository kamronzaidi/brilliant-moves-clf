import os
import torch
import argparse
from shared.utility import *

'''
This script will classify moves as brilliant or not, based on their game trees. Must run generate_trees.py first.
Must have following directory structure:

brilliant_moves_clf
|-- data_extraction.py
|-- models
|   |-- model-7936-2.pth (state_dict for the neural network)
|-- moves
|   |-- move_id_1
|   |   |-- fen.txt (from board before move. This script does not use this file, it is needed for generate_trees.py)
|   |   |-- uci.txt (the move to classify.)
|   |-- move_id_2
        .
        .
        .
|-- trees
|   |-- lc0
|   |   |-- move_id_1
|   |   |   |-- tree_0_1.gml
            .
            .
            .
|   |   |   |-- tree_0_5.gml.txt
|   |-- move_id_2
        .
        .
        .
|   |-- maia
|   |   |-- move_id_1
|   |   |   |-- tree_1_1.gml
            .
            .
            .
|   |   |   |-- tree_1_5.gml.txt
|   |-- move_id_2
        .
        .
        .
'''

def run_inference(X, state_dict = os.path.join('models','model-7936-2.pth'), moves_dir = 'moves'):
    """
    run_inference will print the classifier confidence score and prediction for moves given in X.
    
    :param X: the numpy array containing all the data, in the shape (N, D) for N samples and D features (D=3980).
    :param state_dict: path to the state dict file storing the weights of the neural network.
    :return: None, the output is printed to stdout.
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    h1, h2, h3 = 25, 400, 50 #the hidden layer size hyperparameters in the network
    network = NeuralNetworkDropout(h1,h2,h3,dropout=0.2).to(device)
    network.load_state_dict(torch.load(state_dict, map_location=device))
    network.eval()

    X = torch.tensor(X)

    with torch.no_grad():
        outputs = network(X.to(device).float())
        
    dirs = [i[0] for i in os.walk(moves_dir)][1:]
    for move_num, move in enumerate(dirs):
        move_name = os.path.basename(move)
        # Need to reverse the output to match conventions, because in my training, 0 was brilliant and 1 was non-brilliant.
        print(f'{move_name}: {torch.sigmoid(-outputs[move_num]).item():.2f}, {"Brilliant" if outputs[move_num].item() < 0 else "Not Brilliant"}.')
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'This script will use the neural network to predict if the given moves are brilliant or not')
    parser.add_argument("-d", "--moves_dir", help = "Path to directory containing the move folders.", default = 'moves')
    parser.add_argument("-s", "--state_dict", help = "Path to the state dict file storing the weights of the neural network.", default = os.path.join('models','model-7936-2.pth'))
    args = parser.parse_args()
    
    X = parse_trees(moves_dir=args.moves_dir)
    run_inference(X, state_dict=args.state_dict, moves_dir=args.moves_dir)