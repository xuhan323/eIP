import torch
from torch import nn
from torch_geometric.nn import radius_graph
import pdb
from torch_scatter import scatter
import torch.nn.functional as F
# from SchNet_force_uncer import *
from torch.autograd import grad
from torch.nn import Embedding, Sequential, Linear, Dropout
from torch_geometric.nn import radius_graph
import copy



HARTREE_TO_KCAL_MOL = 627.509
EV_TO_KCAL_MOL = 23.06052
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]  

def radius_graph_pbc(data, radius, max_num_neighbors_threshold, topk_per_pair=None):
        """Computes pbc graph edges under pbc.
        topk_per_pair: (num_atom_pairs,), select topk edges per atom pair
        Note: topk should take into account self-self edge for (i, i)
        """
        atom_pos = data.pos
        num_atoms = data.natoms
        lattice = data.cell
        batch_size = len(num_atoms)
        device = atom_pos.device
        # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
        num_atoms_per_image = num_atoms
        num_atoms_per_image_sqr = (num_atoms_per_image ** 2)

        # index offset between images
        index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
        )

        index_offset_expand = torch.repeat_interleave(
            index_offset, num_atoms_per_image_sqr
        )
        num_atoms_per_image_expand = torch.repeat_interleave(
            num_atoms_per_image, num_atoms_per_image_sqr
        )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
        num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
        index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
        )
        index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
        )
        atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
        )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
        index1 = (torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode='trunc')
        ) + index_offset_expand
        index2 = (
        atom_count_sqr % num_atoms_per_image_expand
        ) + index_offset_expand
    # Get the positions for each atom

        pos1 = torch.index_select(atom_pos, 0, index1)
        pos2 = torch.index_select(atom_pos, 0, index2)
    
        unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
        num_cells = len(unit_cell)
        unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
        )
        unit_cell = torch.transpose(unit_cell, 0, 1)
        unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
        )

    # Compute the x, y, z positional offsets for each cell in each image
        # print(lattice.shape)
        data_cell = torch.transpose(lattice, 1, 2)
        # pdb.set_trace()
        # pbc_offsets = torch.bmm(data_cell.long(), unit_cell_batch.long())
        # print(data_cell.float().shape)
        # print(unit_cell_batch.shape)
        pbc_offsets = torch.bmm(data_cell.float(), unit_cell_batch)
        pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
        )

    # Expand the positions and indices for the 9 cells
        pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
        pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
        index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
        index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
        pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
        atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

        if topk_per_pair is not None:
            assert topk_per_pair.size(0) == num_atom_pairs
            atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
            assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
            atom_distance_sqr_sort_index = (
                atom_distance_sqr_sort_index +
                torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
            topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
            topk_mask = topk_mask.view(-1)
            topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

            topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
            topk_mask.scatter_(0, topk_indices, 1.)
            topk_mask = topk_mask.bool()

        atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
        mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
        index1 = torch.masked_select(index1, mask)
        index2 = torch.masked_select(index2, mask)
        unit_cell = torch.masked_select(
            unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)
        if topk_per_pair is not None:
            topk_mask = torch.masked_select(topk_mask, mask)

        num_neighbors = torch.zeros(len(atom_pos), device=device)
        num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
        num_neighbors = num_neighbors
        max_num_neighbors = torch.max(num_neighbors)

    # Compute neighbors per image
        _max_neighbors = copy.deepcopy(num_neighbors)
        _max_neighbors[
            _max_neighbors > max_num_neighbors_threshold
        ] = max_num_neighbors_threshold
        _num_neighbors = torch.zeros(len(atom_pos) + 1, device=device).long()
        _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
        _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
        _natoms[1:] = torch.cumsum(num_atoms, dim=0)
        num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
        )

        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    # return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image    
    
    # If max_num_neighbors is below the threshold, return early
        if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
        ):
            return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    # atom_distance_sqr.sqrt() distance

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
        distance_sort = torch.zeros(
            len(atom_pos) * max_num_neighbors, device=device
        ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
        index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
        index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
        )
        index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
        )
        distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
        distance_sort = distance_sort.view(len(atom_pos), max_num_neighbors)

    # Sort neighboring atoms based on distance
        distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
        index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
            -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
        mask_within_radius = torch.le(distance_sort, radius * radius)
        index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
        mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
        mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

        if topk_per_pair is not None:
            topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

        edge_index = torch.stack((index2, index1))   
        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask_num_neighbors)
    
        return edge_index, unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    
def get_n_edge(senders, n_node):
        """
        return number of edges for each graph in the batched graph. 
        Has the same shape as <n_node>.
        """
        index_offsets = torch.cat([torch.zeros(1).to(n_node.device), 
                             torch.cumsum(n_node, -1)], dim=-1)
        n_edge = torch.LongTensor([torch.logical_and(senders >= index_offsets[i], 
                                               senders < index_offsets[i+1]).sum() 
                             for i in range(len(n_node))]).to(n_node.device)
        return n_edge

        
def get_pbc_distances(
        pos,
        edge_index,
        lattice,
        cell_offsets,
        num_atoms,
        return_offsets=False,
        return_distance_vec=True,
    ):  
        edge_index = edge_index
        j_index, i_index = edge_index
        num_edges = get_n_edge(j_index, num_atoms)
        distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
        lattice_edges = torch.repeat_interleave(lattice, num_edges, dim=0).float()
        offsets = torch.einsum('bi,bij->bj', cell_offsets.float(), lattice_edges.float())
        distance_vectors += offsets

    # compute distances
        distances = distance_vectors.norm(dim=-1)

        out = {
            "edge_index": edge_index,
            "distances": distances,
        }

        if return_distance_vec:
            out["distance_vec"] = distance_vectors

        if return_offsets:
            out["offsets"] = offsets

        return out


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift
    
def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:
    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class PainnMessage(nn.Module):
    """Message function"""
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff
        
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        
    def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
        # remember to use v_j, s_j but not v_i, s_i        
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)        
        filter_out = filter_weight * scalar_out[edge[:, 1]]
        
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.node_size,
            dim = 1,
        )
        
        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector =  node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)
        
        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        
        return new_node_scalar, new_node_vector

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, node_size: int):
        super().__init__()
        
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vector.shape[-1],                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        return node_scalar + delta_s, node_vector + delta_v



class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin4 = Linear(hidden_channels, 3*1)



        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin4.weight)
        self.lin4.bias.data.fill_(0)
    def forward(self, v):

        v = self.lin1(v)
        v = self.act(v)
        tmp = self.lin4(v)

        f_alpha,f_beta, f_v = tmp.chunk(3, dim=-1)
        
        f_alpha = F.softplus(f_alpha.squeeze(-1)) + 1 + 10e-6
        f_beta = F.softplus(f_beta.squeeze(-1))
        f_v = F.softplus(f_v.squeeze(-1)) + 10e-6
    

        return f_v, f_alpha, f_beta

class PainnModel(nn.Module):
    """PainnModel without edge updating"""
    def __init__(
        self, 
        num_interactions, 
        hidden_state_size, 
        cutoff,
        pdb,
        normalization=True,
        target_mean=[0.0],
        target_stddev=[1.0],
        atomwise_normalization=True, 
        **kwargs,
    ):
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = 20
        self.pdb = pdb        
        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.hidden_state_size, self.edge_embedding_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.hidden_state_size)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
            nn.SiLU(),
            nn.Linear(self.hidden_state_size, 1),
        )
        self.readout_uncertainty = update_u(self.hidden_state_size, self.hidden_state_size)
        # Normalisation constants
        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev[0]))
        self.register_buffer("normalize_mean", torch.tensor(target_mean[0]))

    def forward(self, input_dict, compute_forces=True):
        if compute_forces:
            input_dict.pos.requires_grad_()
        # input_dict.energy = input_dict.y
        z = input_dict.z.long()
        if self.pdb:
            input_dict.cell = input_dict.cell.reshape(-1,3,3)     
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data = input_dict, radius = self.cutoff, max_num_neighbors_threshold = 500
            )
            input_dict.edge_index = edge_index
            input_dict.cell_offsets = cell_offsets
            input_dict.neighbors = neighbors


            out = get_pbc_distances(
                input_dict.pos,
                input_dict.edge_index,
                input_dict.cell,
                input_dict.cell_offsets,
                input_dict.natoms,
            )
            edge = out["edge_index"].t()
            edge_diff = out['distance_vec']
            edge_dist = out["distances"]
        else:
            edge_index = radius_graph(input_dict.pos, r=self.cutoff, batch=input_dict.batch)
            pos = input_dict.pos
            edge = edge_index.t()
            row, col = edge_index
            edge_diff = pos[row] - pos[col]
            edge_dist = edge_diff.norm(dim=-1)

        num_atoms = input_dict.num_atoms = torch.tensor([len(z)]) .to(edge.device)

        
        
        node_scalar = self.atom_embedding(z)
        node_vector = torch.zeros((input_dict.pos.shape[0], 3, self.hidden_state_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype,
                                 )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge, edge_diff, edge_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)
        
        node_scalar = self.readout_mlp(node_scalar)

        f_v, f_alpha, f_beta = self.readout_uncertainty(node_vector)

        energy = scatter(node_scalar, input_dict.batch, dim=0)

        force = -grad(outputs=energy, inputs=input_dict.pos, grad_outputs=torch.ones_like(energy),create_graph=True,retain_graph=True)[0]

        node_para = (force, f_v, f_alpha, f_beta)

        return energy,force,node_para