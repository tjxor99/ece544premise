import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


from treelet import treelet_funct

from dataset import train_dataset, get_token_dict_from_file
import json

# unique_tokens_file = "unique_tokens.json"
# def graph_to_index_offline():
#     """ Assign node names such as Var or VarFunc to unique one_hot vectors (where 1 means [0, 1, 0, ..., 0]).
#     This should be done offline and saved as a dictionary format (json.
    
#     @ Output (.json): Dictionary, with {Node name such as "var", "VarFunc": unique one_hot vector}
#     """
#     count = 0 # Number of unique vars found.
#     NUM_TOKENS = 1909
#     node_to_index = {}

#     for datapoint in train_dataset():
#         conjecture = datapoint.conjecture
#         statement = datapoint.statement
#         for _, node in conjecture.nodes.items():
#             token = node.token
#             if token not in node_to_index:
#                 node_to_index[token] = count
#                 count += 1
#                 print("Number of tokens found: "+str(count))
#             if count == NUM_TOKENS:
#                 print("All Unique Tokens Found!")
#                 dumped = json.dumps(node_to_index)
#                 with open(unique_tokens_file, "w") as f:
#                     f.write(dumped)
#                 return


class LinearMap(nn.Module):
    '''
    Map the one-hot vector to the x's corresponding to inputs of each neural network.
    '''
    def __init__(self, INPUT_DIM = 1909):
        super(LinearMap, self).__init__()
        # INPUT_DIM = 1909
        HL1 = 256
        self.fc1 = nn.Linear(INPUT_DIM, HL1)

    def forward(self, x):
        return self.fc1(x)



class FPClass(nn.Module):
    '''
    FP is the outer function in Section 3.3 used for Order-Preserving Embeddings 
    Input:  Linear combination of the node and neighboring update functions.
    Output: The next value for the node.
    '''
    def __init__(self):
        super(FPClass, self).__init__()
        INPUT_DIM = 256
        HL1 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        return x

class FIClass(nn.Module):
    def __init__(self):
        super(FIClass, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        """
        @ Args:
            x_batch (dense vectors, shape = [batch_size, length of parents(xv), xv for each xv)
                Collection of (x_u, x_v) 

                x_batch[:,:, xv_id] = The summands of FI for fixed xv

        @ Output:
            in_sum (array of dense_vectors): Size = Number of Nodes in G
        """
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch



class FHClass(nn.Module):
    def __init__(self):
        super(FHClass, self).__init__()
        INPUT_DIM = 256 * 3
        HL1 = HL2 = 256        
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)
        self.fc2 = nn.Linear(HL1, HL2)
        self.bn2 = nn.BatchNorm1d(HL2)

    def forward(self, x_batch):
        # Order in xv, xu, xw
        x_batch = F.relu(self.bn1(self.fc1(x_batch)))
        x_batch = F.relu(self.bn2(self.fc2(x_batch)))
        return x_batch




class CondClassifier(nn.Module):
    def __init__(self):
        super(CondClassifier, self).__init__()
        INPUT_DIM = 256 * 2
        HL1 = 256
        HL2 = 2
        self.fc1 = nn.Linear(INPUT_DIM, HL1)
        self.bn1 = nn.BatchNorm1d(HL1)

        self.fc2 = nn.Linear(HL1, HL2)

    def forward(self, x_conj, x_state):
        x = torch.cat([x_conj, x_state], dim = 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)

        return x

    def predict(self, x_conj, x_state): # Only used for testing.
        scores = self.forward(x_conj, x_state)
        return np.argmax(scores)


class max_pool_dense_graph(nn.Module):
    def __init__(self):
        super(max_pool_dense_graph, self).__init__()
        INPUT_DIM = 256 * 2
        self.pool = nn.MaxPool1d(2)

    def forward(self, G_dense):
        x1 = G_dense[0]
        x1.unsqueeze_(0)
        for x2 in G_dense:
            x2.unsqueeze_(0)
            x = torch.stack([x1, x2], dim = 2)

            x1 = self.pool(x)
            x1.squeeze_(2)
        x1.squeeze_(0)
        return x1


class FormulaNet(nn.Module):
    def __init__(self, num_steps, cuda_available = False):
        super(FormulaNet, self).__init__()
        # Initialize models
        self.token_to_index = get_token_dict_from_file()
        self.num_tokens = len(self.token_to_index)

        self.dense_map = LinearMap(self.num_tokens) # maps one_hot -> 256 dimension vector
        self.FP = FPClass()
        self.FI = FIClass()
        self.FO = FIClass()
        self.FL = FHClass()
        self.FR = FHClass()
        self.FH = FHClass()
        self.Classifier = CondClassifier()

        self.max_pool_dense_graph = max_pool_dense_graph()

        self.num_steps = num_steps
        self.cuda_available = cuda_available


    # Given graphs and all the functions, do one update in parallel for all nodes in the graph.
    def fullPass(self, dense_nodes, Gs):
        """
        @ Args:
            dense_nodes (3D Array): 0-axis is the inter-graph batching index
                dense_nodes[batch_index][i] is the 256 dimensional dense representation of the ith node, 
                where i is the node's unique id in <Graph> object
            G (Array of <Graph> object): Graph object, necessary to iterate over edges and trelets

        @ Vars:
            in_batch (list): Whatever will be fed into FI, FO, ... FR
            dv (Tensor): dv[xv] for each xv
            ev (Tensor): ev[xv] for each xv
            <Neural Function>_indices (dict): {xv: [index of in_batch]} to be used to for summands in update equation (indices of in_sum).
            new_<Neural Function>_sum (Tensor): in_sum, out_sum, etc. collapsed such that new_sum[xv] = sum_{xu in parent(xv)} F_I (xu, xv)
                shape: (num_nodes, 256)

        @ Return:
            new_nodes (Same shape as Mdense_nodes>): One-step update of <dense_nodes>.
        """
        # dv is determined by the number of summands for FI + summands of FO
        if self.cuda_available:
            dv = torch.zeros([dense_nodes.shape[0]]).cuda()
            ev = torch.zeros([dense_nodes.shape[0]]).cuda()
        else:
            dv = torch.zeros(dense_nodes.shape[0])
            ev = torch.zeros([dense_nodes.shape[0]])

        start_index = 0 # To keep track of which graph's xv_id and xu_id we are using.

        in_index = 0
        in_indices = {} # for each x in Gs, in_indices[x] gives the indices of in_batch related for summing
        in_batch = []

        out_index = 0
        out_batch = []
        out_indices = {}

        left_index = 0
        left_batch = []
        left_indices = {}

        head_index = 0
        head_batch = []
        head_indices = {}

        right_index = 0
        right_batch = []
        right_indices = {}

        for G in Gs: # Inter-graph Batching.
            end_index = start_index + len(G.nodes)

            treelets = treelet_funct(G) # Treelets for this graph.

            for xv_id, xv_obj in G.nodes.items():
                xv_id_offset = xv_id + start_index
                xv_dense = dense_nodes[xv_id_offset]

                in_indices[xv_id_offset] = [] # indices of in_batch corresponding to each (xu, xv) pair, forall xu
                out_indices[xv_id_offset] = []

                for xu_id in xv_obj.parents: # For FI
                    xu_id_offset = xu_id + start_index
                    in_indices[xv_id_offset].append(in_index)
                    in_index += 1

                    xu_dense = dense_nodes[xu_id_offset]
                    in_batch.append(torch.cat([xu_dense, xv_dense], dim = 0))
                    dv[xv_id_offset] += 1

                for xu_id in xv_obj.children: # For FO
                    xu_id_offset = xu_id + start_index
                    out_indices[xv_id_offset].append(out_index)
                    out_index += 1

                    xu_dense = dense_nodes[xu_id_offset]
                    out_batch.append(torch.cat([xv_dense, xu_dense], dim = 0))
                    dv[xv_id_offset] += 1

                # ----------------------- Iterate over treelets ----------------------- #
                # Left Treelet: (xv, xu, xw)
                left_indices[xv_id_offset] = []
                for _, xu_id, xw_id in treelets[xv_id][0]:
                    xu_id_offset = xu_id + start_index
                    xw_id_offset = xw_id + start_index

                    left_indices[xv_id_offset].append(left_index)
                    left_index += 1

                    xu_dense = dense_nodes[xu_id_offset]
                    xw_dense = dense_nodes[xw_id_offset]
                    left_batch.append(torch.cat([xv_dense, xu_dense, xw_dense], dim = 0))
                    ev[xv_id_offset] += 1

                # Head Treelet: (xu, xv, xw)
                head_indices[xv_id_offset] = []
                for xu_id, _, xw_id in treelets[xv_id][1]:
                    xu_id_offset = xu_id + start_index
                    xw_id_offset = xw_id + start_index

                    head_indices[xv_id_offset].append(head_index)
                    head_index += 1

                    xu_dense = dense_nodes[xu_id_offset]
                    xw_dense = dense_nodes[xw_id_offset]
                    head_batch.append(torch.cat([xu_dense, xv_dense, xw_dense]))
                    ev[xv_id_offset] += 1

                # Right Treelet: (xu, xw, xv)
                right_indices[xv_id_offset] = []
                for xu_id, xw_id, _ in treelets[xv_id][2]:
                    xu_id_offset = xu_id + start_index
                    xw_id_offset = xw_id + start_index

                    right_indices[xv_id_offset].append(right_index)
                    right_index += 1

                    xu_dense = dense_nodes[xu_id_offset]
                    xw_dense = dense_nodes[xw_id_offset]
                    right_batch.append(torch.cat([xu_dense, xw_dense, xv_dense]))
                    ev[xv_id_offset] += 1

            start_index += len(G.nodes)

        if len(in_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            # print(1)
            if self.cuda_available:
                in_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                in_sum = torch.zeros(dense_nodes.shape)
        else:
            in_batch = torch.stack(in_batch, dim = 0)
            in_sum = self.FI(in_batch) 

        # Pass out_batch into FO
        if len(out_batch) <= 1: 
            # print(2)
            if self.cuda_available:
                out_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                out_sum = torch.zeros(dense_nodes.shape)
        else:
            out_batch = torch.stack(out_batch, dim = 0)
            out_sum = self.FO(out_batch)

        # Left Treelet
        if len(left_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            # print(3)
            if self.cuda_available:
                left_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                left_sum = torch.zeros(dense_nodes.shape)
        else:
            left_batch = torch.stack(left_batch, dim = 0)
            left_sum = self.FL(left_batch)


        # Head Treelet
        if len(head_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            # print(4)
            if self.cuda_available:
                head_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                head_sum = torch.zeros(dense_nodes.shape)
        else:
            head_batch = torch.stack(head_batch, dim = 0)
            head_sum = self.FH(head_batch)


        # Right Treelet
        if len(right_batch) <= 1: # When the batch size is 1, we can't do batch normalization. Skip
            # print(5)
            if self.cuda_available:
                right_sum = torch.zeros(dense_nodes.shape).cuda()
            else:
                right_sum = torch.zeros(dense_nodes.shape)
        else:
            right_batch = torch.stack(right_batch, dim = 0)
            right_sum = self.FR(right_batch)



# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# Does assigning to in_out_sum and treelet_sum maintain the computation graph?
# I'm assuming it does, since the elements in in_out_sum and treelet_sum are all Tensor objects.
# ----------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------- #
        # Compute in_out_sum and treelet_sum to input into FP

        if self.cuda_available:
            in_out_sum = torch.empty(dense_nodes.shape).cuda() 
            treelet_sum = torch.empty(dense_nodes.shape).cuda()
        else:
            in_out_sum = torch.empty(dense_nodes.shape)
            treelet_sum = torch.empty(dense_nodes.shape)

        start_index = 0
        for G in Gs:
            end_index = start_index + len(G.nodes)
            for xv_id in G.nodes.keys():
                xv_id_offset = xv_id + start_index

                if dv[xv_id_offset] == 0:
                    if self.cuda_available:
                        in_out_sum[xv_id_offset] = torch.zeros(256).cuda()
                    else:
                        in_out_sum[xv_id_offset] = torch.zeros(256)
                else:
                    new_in_sum = torch.sum(in_sum[in_indices[xv_id_offset], :], dim = 0)
                    new_out_sum = torch.sum(out_sum[out_indices[xv_id_offset], :], dim = 0)
                    in_out_sum[xv_id_offset] = (new_in_sum + new_out_sum) / dv[xv_id_offset]

                if ev[xv_id_offset] == 0:
                    if self.cuda_available:
                        treelet_sum[xv_id_offset] = torch.zeros(256).cuda()
                    else:
                        treelet_sum[xv_id_offset] = torch.zeros(256)
                else:
                    new_left_sum = torch.sum(left_sum[left_indices[xv_id_offset], :], dim = 0)
                    new_head_sum = torch.sum(head_sum[head_indices[xv_id_offset], :], dim = 0)
                    new_right_sum = torch.sum(right_sum[right_indices[xv_id_offset], :], dim = 0)
                    treelet_sum[xv_id_offset] = (new_left_sum + new_head_sum + new_right_sum) / ev[xv_id_offset]

            start_index += len(G.nodes)

        # Add and then send to FP to update all the nodes!
        new_nodes = self.FP(dense_nodes + in_out_sum + treelet_sum)

        return new_nodes

    def graph_to_one_hot(self, G):
        """
        Given a graph object, return an array of one-hot vectors.
        """
        NUM_TOKENS = self.num_tokens
        if self.cuda_available:
            one_hot_graph = torch.zeros((len(G.nodes), NUM_TOKENS)).cuda()
        else:
            one_hot_graph = torch.zeros((len(G.nodes), NUM_TOKENS))

        index = 0
        for _, node in G.nodes.items():
            token = node.token
            if token not in self.token_to_index:
                token = "UNKNOWN"
            token_index = self.token_to_index[token]

            one_hot_graph[index, token_index] = 1
        return one_hot_graph

    def forward(self, conjecture_state_graphs):
        """
        @ Args:
            conjecture_graph (arr-like of Graph Objects): Inter-graph Batch of Conjectures
            statement_graph (same)
        """
        inter_graph_conj_state_node_batch = []
        conj_state_graphs = []

        start_index = 0
        conj_indices = {} # conj_index[i] = [first node in conjecture i, last node in conjecture i]
        state_indices = {} # similar
        for i in range(len(conjecture_state_graphs)): # Iterate over inter-graph-batch.
            conjecture_graph = conjecture_state_graphs[i][0]
            end_index = start_index + len(conjecture_graph.nodes)
            conj_indices[i] = [start_index, end_index]

            start_index = end_index
            statement_graph = conjecture_state_graphs[i][1]
            end_index = start_index + len(statement_graph.nodes)
            state_indices[i] = [start_index, end_index]

            # Map graph object to an array of one hot vectors
            conj_one_hot = self.graph_to_one_hot(conjecture_graph)
            state_one_hot = self.graph_to_one_hot(statement_graph)

            conj_state_node_batch = torch.cat([conj_one_hot, state_one_hot], dim = 0)
            inter_graph_conj_state_node_batch.append(conj_state_node_batch)

            conj_state_graphs.append(conjecture_graph)
            conj_state_graphs.append(statement_graph)

        inter_graph_conj_state_node_batch = torch.cat(inter_graph_conj_state_node_batch, dim = 0) # [:, 1909] Tensor (as if all nodes belonged to one huge graph)
        conj_state_dense_batch = self.dense_map(inter_graph_conj_state_node_batch)

        # Iterate equation 2.
        for t in range(self.num_steps):
            conj_state_dense_batch = self.fullPass(conj_state_dense_batch, conj_state_graphs)
            
        # Finished Updating. max-pool over all nodes in the graph

        # -------------------------- This will have to be modified ---------------------------- #
        # max_pool across each relevant graph. For example, the first max-pool should be over the first 36 nodes.
        conj_embeddings = []
        state_embeddings = []
        for i in range(len(conj_indices)):
            conj_embeddings.append(self.max_pool_dense_graph(conj_state_dense_batch[conj_indices[i][0] : conj_indices[i][1]]))
            state_embeddings.append(self.max_pool_dense_graph(conj_state_dense_batch[state_indices[i][0] : state_indices[i][1]]))

        conj_embeddings = torch.stack(conj_embeddings)
        state_embeddings = torch.stack(state_embeddings)

        # Classify
        # -------------------------- This will have to be modified ---------------------------- #
        # Classify across each conjecture-state embeddings. (for example, each should take only 2 indices each.)
        prediction = self.Classifier(conj_embeddings, state_embeddings)

        return prediction



# if __name__ == "__main__":
    # graph_to_index_offline()
