import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
import argparse
from datasets.pcpnet_dataset import PCPNetDataset
from torch_geometric.data import DataLoader
from utils.radius import radius_graph

from torch_sym3eig import Sym3Eig
from torch_geometric.data import Data

import os
from networks.gnn import GNNFixedK
from utils.covariance import compute_cov_matrices_dense, compute_weighted_cov_matrices_dense

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', default=None, help='Path where results (normals) are stored')
parser.add_argument('--model_name', default='network_k64.pt', help='Model file from trained_models/ to use')
parser.add_argument('--dataset_path', type=str, default='data/pcpnet_data/', help='Path at which dataset is created')
parser.add_argument('--k_test', type=int, default=64, help='Neighborhood size for eval [default: 64]')
parser.add_argument('--iterations', type=int, default=4, help='Number of iterations for testing [default: 4]')
FLAGS = parser.parse_args()

if FLAGS.results_path is not None:
    if not os.path.exists(FLAGS.results_path):
        os.makedirs(FLAGS.results_path)

path = 'data/pcpnet_data/'

transform = T.Compose([T.NormalizeScale()])




# Replace this with the path to your .xyz file
xyz_file_path = FLAGS.dataset_path

# Load the .xyz data and convert it to a torch_geometric.data.Data object
def xyz_to_data(file_path):
    points = np.loadtxt(file_path)
    pos = torch.tensor(points, dtype=torch.float32)
    data = Data(pos=pos)
    return data

# Custom dataset loader for your .xyz file
def custom_loader(file_path):
    data = xyz_to_data(file_path)
    return DataLoader([data], batch_size=1)



def save_normals(normals):
    #category_file = category_files_test[test_set]
    #file_path = osp.join(path, 'raw', category_file)
    #with open(file_path, "r") as f:
        #filenames = f.read().split('\n')[:-1]
    #file = filenames[example]
    #data/custom_data/cup.xyz
    filenames = xyz_file_path.split('/')
    name2 = filenames[2]
    file = name2.split(".")[0]
    out_path = osp.join(FLAGS.results_path, file+'.normals')
    normals = normals.cpu().numpy()
    np.savetxt(out_path, normals, delimiter=' ')

test_custom_loader = custom_loader(xyz_file_path)

category_files_test = ['testset_no_noise.txt',
        'testset_low_noise.txt', 'testset_med_noise.txt',
        'testset_high_noise.txt', 'testset_vardensity_striped.txt',
        'testset_vardensity_gradient.txt']




# Normal estimation algorithm
# forward() corresponds to one iteration of Algorithm 1 in the paper
class NormalEstimation(torch.nn.Module):
    def __init__(self):
        super(NormalEstimation, self).__init__()
        self.stepWeights = GNNFixedK()

    def forward(self, old_weights, pos, batch, normals, edge_idx_l, dense_l, stddev):
        # Re-weighting
        weights = self.stepWeights(pos, old_weights, normals, edge_idx_l, dense_l, stddev)  # , f=f)

        # Weighted Least-Squares
        cov = compute_weighted_cov_matrices_dense(pos, weights, dense_l, edge_idx_l[0])
        eig_val, eig_vec = Sym3Eig.apply(cov)
        _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
        eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
        normals = eig_vec[:, :, 0]

        # Not necessary for PCPNetDataset but might be for other datasets with underdefined neighborhoods
        # mask = torch.isnan(normals)
        # normals[mask] = 0.0

        return normals, weights


device = torch.device('cuda')
model = NormalEstimation().to(device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('num_params:', params)


def test(loader, string, test_set, size):
    model.eval()
    print('Starting eval: {}, k_test = {}, Iterations: {} '.format(string, size, FLAGS.iterations))

    with torch.no_grad():
        for i, data in enumerate(loader):
            pos, batch = data.pos, data.batch

            # Compute statistics for normalization
            edge_idx_16, _ = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=16)
            row16, col16 = edge_idx_16
            cart16 = (pos[col16].cuda() - pos[row16].cuda())
            stddev = torch.sqrt((cart16 ** 2).mean()).detach().item()

            # Compute KNN-graph indices for GNN
            edge_idx_l, dense_l = radius_graph(pos, 0.5, batch=batch, max_num_neighbors=size)

            # Iteration 0 (PCA)
            cov = compute_cov_matrices_dense(pos, dense_l, edge_idx_l[0]).cuda()
            eig_val, eig_vec = Sym3Eig.apply(cov)
            _, argsort = torch.abs(eig_val).sort(dim=-1, descending=False)
            eig_vec = eig_vec.gather(2, argsort.view(-1, 1, 3).expand_as(eig_vec))
            normals = eig_vec[:, :, 0]
            edge_idx_c = edge_idx_l.cuda()
            pos, batch = pos.detach().cuda(), batch.detach().cuda()
            old_weights = torch.ones_like(edge_idx_c[0]).float() / float(size)

            # Loop of Algorithm 1 in the paper
            for j in range(FLAGS.iterations):
                normals, old_weights = model(old_weights.detach(), pos, batch, normals.detach(),
                                                                     edge_idx_c, edge_idx_c[1].view(pos.size(0), -1), stddev)
            if FLAGS.results_path is not None:
                save_normals(normals)
            # You can access and process the computed normals here
            print("Normals computed for batch #{}".format(i))


def run():
    size = FLAGS.k_test
    test(test_custom_loader, 'Custom', 0, size)


model.load_state_dict(torch.load('trained_models/{}'.format(FLAGS.model_name)))
run()
