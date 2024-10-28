import torch
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import torch.nn.functional as F
from torch.nn import (
    Sequential,
    Linear,
    ReLU,
    Dropout,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger
import time


from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
)

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import argparse
import os
from torch import nn
import contextlib

import torch.cuda
from pytorch_lightning.callbacks import Callback


class DTIDataset(Dataset):
    def __init__(self, protein_embeddings, df, mean_agg=False, cls=False):
        self.protein_embeddings = protein_embeddings.copy()
        self.mean_agg = mean_agg

        for k, v in protein_embeddings.items():
            if mean_agg:
                self.protein_embeddings[k] = torch.mean(v[1:-1], dim=0)
            elif cls:
                self.protein_embeddings[k] = v[0]
            else:
                self.protein_embeddings[k] = v[1:-1]

        seqs = df["Target Sequence"].tolist()

        labels = df["Label"].tolist()
        self.drug_target_interactions = [
            (seq.upper(), label)
            for seq,  label in zip(seqs, labels)
        ]

    def __len__(self):
        return len(self.drug_target_interactions)

    def __getitem__(self, idx):
        seq, label = self.drug_target_interactions[idx]

        protein_embedding = self.protein_embeddings[seq]

        return (
            protein_embedding,
            torch.tensor(label, dtype=protein_embedding.dtype),
        )


def non_attention_collate_fn(batch):
    protein_embeddings, labels = zip(*batch)
    
    protein_embeddings, labels = (
        torch.stack(protein_embeddings),

        torch.stack(labels)
    )

    return protein_embeddings, labels, None


def attention_collate_fn(batch):
    protein_embeddings,  labels = zip(*batch)

    max_seq_len = max([seq.shape[0] for seq in protein_embeddings])

    padded_protein_embeddings = []
    mask = []
    for seq in protein_embeddings:

        padded_seq = F.pad(seq, (0, 0, 0, max_seq_len - len(seq)), value=0)
        padded_protein_embeddings.append(padded_seq)

        seq_mask = torch.ones(len(seq), dtype=torch.float)
        padded_mask = F.pad(seq_mask, (0, max_seq_len - len(seq)), value=0)
        mask.append(padded_mask)

    padded_protein_embeddings,  labels, mask = (
        torch.stack(padded_protein_embeddings),
     
        torch.stack(labels),
        torch.stack(mask),
    )

    return padded_protein_embeddings, labels, mask.bool()


class DTIDataModule(L.LightningDataModule):
    def __init__(
        self,
        protein_embeddings,
        train_df,
        val_df,
        test_df,
        collate_fn,
        num_workers=14,
        batch_size=32,
        mean_agg=False,
        cls=False,
    ):

        if mean_agg and cls:
            raise Exception("Only one of mean_agg or cls should be True")

        self.train_dataset = DTIDataset(protein_embeddings, train_df, mean_agg, cls)
        self.val_dataset = DTIDataset(protein_embeddings, val_df, mean_agg, cls)
        self.test_dataset = DTIDataset(protein_embeddings, test_df, mean_agg, cls)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


#####
# Sliced-Wasserstein embedding code below (Interp1d and SWE_Pooling classes) from "Aggregating Residue-Level Protein Language Model Embeddings with Optimal Transport" by Navid NaderiAlizadeh and Rohit Singh (https://www.biorxiv.org/content/10.1101/2024.01.29.577794v1.full.pdf, https://github.com/navid-naderi/PLM_SWE)
#####
class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class SWE_Pooling(nn.Module):
    def __init__(self, d_in, num_ref_points, num_slices):
        '''
        Produces fixed-dimensional permutation-invariant embeddings for input sets of arbitrary size based on sliced-Wasserstein embedding.
        Inputs:
            d_in:  The dimensionality of the space that each set sample belongs to
            num_ref_points: Number of points in the reference set
            num_slices: Number of slices
        '''
        super(SWE_Pooling, self).__init__()
        self.d_in = d_in
        self.num_ref_points = num_ref_points
        self.num_slices = num_slices

        uniform_ref = torch.linspace(-1, 1, num_ref_points).unsqueeze(1).repeat(1, num_slices)
        self.reference = nn.Parameter(uniform_ref)

        self.theta = nn.utils.weight_norm(nn.Linear(d_in, num_slices, bias=False), dim=0)
        if num_slices <= d_in:
            nn.init.eye_(self.theta.weight_v)
        else:
            nn.init.normal_(self.theta.weight_v)
            
        self.theta.weight_g.data = torch.ones_like(self.theta.weight_g.data, requires_grad=False)
        self.theta.weight_g.requires_grad = False

        # weights to reduce the output embedding dimensionality
        self.weight = nn.Linear(num_ref_points, 1, bias=False)

    def forward(self, X, mask=None):
        '''
        Calculates GSW between two empirical distributions.
        Note that the number of samples is assumed to be equal
        (This is however not necessary and could be easily extended
        for empirical distributions with different number of samples)
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
            mask [optional]: B x N binary tensor, with 1 iff the set element is valid; used for the case where set sizes are different
        Output:
            weighted_embeddings: B x num_slices tensor, containing a batch of B embeddings, each of dimension "num_slices" (i.e., number of slices)
        '''

        B, N, _ = X.shape       
        Xslices = self.get_slice(X)

        M, _ = self.reference.shape

        if mask is None:
            # serial implementation should be used if set sizes are different
            Xslices_sorted, Xind = torch.sort(Xslices, dim=1)

            if M == N:
                Xslices_sorted_interpolated = Xslices_sorted
            else:
                x = torch.linspace(0, 1, N + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
                xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
                y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_slices, -1)
                Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2)
        else:
            # replace invalid set elements with points to the right of the maximum element for each slice and each set (which will not impact the sorting and interpolation process)
            invalid_elements_mask = ~mask.bool().unsqueeze(-1).repeat(1, 1, self.num_slices)
            Xslices_copy = Xslices.clone()
            Xslices_copy[invalid_elements_mask] = -1e10

            top2_Xslices, _ = torch.topk(Xslices_copy, k=2, dim=1)
            max_Xslices = top2_Xslices[:, 0].unsqueeze(1)
            delta_y = - torch.diff(top2_Xslices, dim=1)

            Xslices_modified = Xslices.clone()

            Xslices_modified[invalid_elements_mask] = max_Xslices.repeat(1, N, 1)[invalid_elements_mask]

            delta_x = 1 / (1 + torch.sum(mask, dim=1, keepdim=True))
            slope = delta_y / delta_x.unsqueeze(-1).repeat(1, 1, self.num_slices) # B x 1 x num_slices
            slope = slope.repeat(1, N, 1)

            eps = 1e-3
            x_shifts = eps * torch.cumsum(invalid_elements_mask, dim=1)
            y_shifts = slope * x_shifts
            Xslices_modified = Xslices_modified + y_shifts

            Xslices_sorted, _ = torch.sort(Xslices_modified, dim=1)

            x = torch.arange(1, N + 1).to(X.device) / (1 + torch.sum(mask, dim=1, keepdim=True)) # B x N

            invalid_elements_mask = ~mask.bool()
            x_copy = x.clone()
            x_copy[invalid_elements_mask] = -1e10
            max_x, _ = torch.max(x_copy, dim=1, keepdim=True)
            x[invalid_elements_mask] = max_x.repeat(1, N)[invalid_elements_mask]

            x = x.unsqueeze(1).repeat(1, self.num_slices, 1) + torch.transpose(x_shifts, 1, 2)
            x = x.view(-1, N) # BL x N

            xnew = torch.linspace(0, 1, M + 2)[1:-1].unsqueeze(0).repeat(B * self.num_slices, 1).to(X.device)
            y = torch.transpose(Xslices_sorted, 1, 2).reshape(B * self.num_slices, -1)
            Xslices_sorted_interpolated = torch.transpose(Interp1d()(x, y, xnew).view(B, self.num_slices, -1), 1, 2)

        Rslices = self.reference.expand(Xslices_sorted_interpolated.shape)

        _, Rind = torch.sort(Rslices, dim=1)
        embeddings = (Rslices - torch.gather(Xslices_sorted_interpolated, dim=1, index=Rind)).permute(0, 2, 1) # B x num_slices x M

        weighted_embeddings = self.weight(embeddings).sum(-1)

        return weighted_embeddings.view(-1, self.num_slices)

    def get_slice(self, X):
        '''
        Slices samples from distribution X~P_X
        Input:
            X:  B x N x dn tensor, containing a batch of B sets, each containing N samples in a dn-dimensional space
        '''
        return self.theta(X)
######

class ModelModule(L.LightningModule):
    def __init__(
        self,
        num_steps,
        module_type,
        learning_rate,
        protein_embedding_dim,
    
        attention_model_dim,
        classification,
        num_attention_heads=8
    ):
        super().__init__()

        self.num_steps = num_steps * 10
        self.module_type = module_type

        if module_type in {"attention_aggregation", "multihead_attention_aggregation"}:
            input_dim = attention_model_dim
        else:
            input_dim = protein_embedding_dim 

        self.similarity_predictor = Sequential(
            Linear(input_dim, input_dim // 4),
            ReLU(),
            Dropout(0.5),
            Linear(input_dim // 4, 1),
        )

        if self.module_type in {
            "attention_aggregation",
            "multihead_attention_aggregation",
        }:
            if self.module_type == "attention_aggregation":
                self.pool_query = torch.nn.Parameter(
                    torch.ones(1, attention_model_dim).to(self.device)
                )
            else:
                self.pool_query = torch.nn.Parameter(torch.ones(num_attention_heads, 1, attention_model_dim //num_attention_heads, device=self.device))

            self.k_proj = Linear(protein_embedding_dim, attention_model_dim)
            self.v_proj = Linear(protein_embedding_dim, attention_model_dim)

        if self.module_type == "swe":
            self.swe = SWE_Pooling(input_dim, 128, 1024)

            self.target_projector = nn.Sequential(
            nn.Linear(1024, input_dim), nn.ReLU()
        )

        if classification:
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.MSELoss()

        self.lr = learning_rate
        self.num_attention_heads = num_attention_heads

    def split_heads(self, x, batch_size):
        head_dim = x.shape[-1] // self.num_attention_heads
        x = x.view(batch_size, -1, self.num_attention_heads, head_dim)
        return x.transpose(1, 2)

    def aggregate(self, protein_embeddings, mask, dropout_prob=0.1, multihead=False):
        batch_size = protein_embeddings.shape[0]

        k = self.k_proj(protein_embeddings)
        v = self.v_proj(protein_embeddings)

        if multihead:
            q = self.pool_query.expand(batch_size, -1, -1, -1)

            k = self.split_heads(
                k, batch_size
            ) 
            v = self.split_heads(
                v, batch_size
            ) 

            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / (
                k.shape[-1] ** 0.5
            )
            mask = mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = F.dropout(
                attention_weights, p=dropout_prob, training=self.training
            )

            aggregated_protein_embeddings = torch.matmul(attention_weights, v)

            aggregated_protein_embeddings = aggregated_protein_embeddings.transpose(
                1, 2
            ).contiguous()
            aggregated_protein_embeddings = aggregated_protein_embeddings.view(
                batch_size, 1, -1
            ).squeeze(1)

        else:
            q = self.pool_query.expand(len(protein_embeddings), -1, -1)

            attention_scores = torch.bmm(q, k.transpose(1, 2))
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

            attention_weights = F.softmax(
                attention_scores / (k.shape[-1] ** 0.5), dim=-1
            )
            attention_weights = F.dropout(
                attention_weights, p=dropout_prob, training=self.training
            )

            aggregated_protein_embeddings = torch.bmm(attention_weights, v).squeeze(1)

        return aggregated_protein_embeddings, attention_weights

    def mlp_similarity(
        self, protein_embeddings, mask, return_attention_weights=False
    ):
        if self.module_type in {
            "attention_aggregation",
            "multihead_attention_aggregation",
        }:
            if self.module_type == "multihead_attention_aggregation":
                multihead = True
            else:
                multihead = False
            protein_embeddings, attention_weights = self.aggregate(
                protein_embeddings, mask, multihead=multihead
            )

        if self.module_type == "swe":
            protein_embeddings = self.target_projector(self.swe(protein_embeddings, mask))
            

        similarity = self.similarity_predictor(protein_embeddings)

        if return_attention_weights:
            return similarity.squeeze(1), attention_weights.squeeze(1)
        else:
            return similarity.squeeze(1)

    def training_step(self, batch, batch_idx):

        padded_protein_embeddings, labels, mask = batch

        similarity = self.mlp_similarity(padded_protein_embeddings,  mask)

        loss = self.loss(similarity, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):


        padded_protein_embeddings, labels, mask = batch

        similarity = self.mlp_similarity(padded_protein_embeddings,  mask)

        loss = self.loss(similarity, labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):

        padded_protein_embeddings, labels, mask = batch

        similarity = self.mlp_similarity(padded_protein_embeddings,  mask)

        loss = self.loss(similarity, labels)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):

        padded_protein_embeddings,labels, mask = batch

        similarity = self.mlp_similarity(
            padded_protein_embeddings,
 
            mask,
            return_attention_weights=False,
        )
        return similarity, labels


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)


        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.num_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "name": "learning_rate",
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def calc_metrics(y_true, y_score, classification=True):
    y_true, y_score = y_true.float().cpu().detach().numpy(), y_score.float().cpu().detach().numpy()
    if classification:
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(recall, precision)

        auroc = roc_auc_score(y_true, y_score)

        y_pred = (y_score > 0.5).astype(int) 
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return auroc, auprc, f1, mcc
    
    else:
        corr = np.corrcoef(np.array([y_true, y_score]))[0, 1]
        print(y_true.shape, y_score.shape)
        rmse = np.sqrt(np.mean(np.square(y_true - y_score))) 

        return corr, rmse
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_directory", type=str, required=True)
    parser.add_argument("--embedding_dict_file_name", type=str, required=True)
    parser.add_argument("--protein_embedding_dimension", type=int, required=True)
    parser.add_argument("--module_type", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument(
        "--classification_task",
        type=lambda x: (str(x).lower() == "true"),
        required=True,
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()

    train_df = pd.read_pickle(os.path.join(args.data_directory, "train.pkl"))
    val_df = pd.read_pickle(os.path.join(args.data_directory, "val.pkl"))
    test_df = pd.read_pickle(os.path.join(args.data_directory, "test.pkl"))


    with open(os.path.join(args.data_directory, args.embedding_dict_file_name), "rb") as f:
        protein_embeddings = pickle.load(f)

    if args.classification_task:
        auroc_list, auprc_list, f1_list, mcc_list = [], [], [], []
    else:
        correlation_list, rmse_list = [], []

    if args.module_type in {"attention_aggregation", "multihead_attention_aggregation", "swe"}:
        collate_fn = attention_collate_fn
    else:
        collate_fn = non_attention_collate_fn

    mean_agg, cls = False, False
    if args.module_type == "mean_aggregation":
        mean_agg = True

    if args.module_type == "cls":
        cls = True


    for i in range(5):
        torch.manual_seed(i)
        datamodule = DTIDataModule(
            protein_embeddings,
            train_df,
            val_df,
            test_df,
            collate_fn,
            num_workers=8,
            batch_size=args.batch_size,
            mean_agg=mean_agg,
            cls=cls,
        )

        num_steps = (
            len(datamodule.train_dataset.drug_target_interactions) // args.batch_size
        )
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        max_epochs = 100

        model = ModelModule(
            num_steps,
            module_type=args.module_type,
            learning_rate=args.learning_rate,
            protein_embedding_dim=args.protein_embedding_dimension,
            attention_model_dim=args.protein_embedding_dimension,
            classification=args.classification_task,
        )

        model_name = f"{args.model_name}_{args.module_type}_run_{i}"
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_weights_only=True, 
            dirpath=args.data_directory,
            save_top_k=1,
            filename=f"{args.model_name}-{args.module_type}-run-{i}-lr-{args.learning_rate}-batch_size-{datamodule.batch_size}-"
            + "{epoch}-{val_loss:.2f}",
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='step')
        logger = TensorBoardLogger("lightning_logs", name=model_name)
        trainer = L.Trainer(
            max_epochs=max_epochs,
            devices=1,
            accelerator="gpu",
            log_every_n_steps=100,
            precision="bf16",
            logger=logger,
            callbacks=[checkpoint_callback, lr_monitor],
        )
        trainer.fit(model, train_loader, val_loader)


        test_loader = datamodule.test_dataloader()
        predictions = trainer.predict(dataloaders=test_loader, ckpt_path="best")
        
        score, labels = zip(*predictions)
        score, labels = torch.cat(score), torch.cat(labels)     

        if args.classification_task:
            score = torch.sigmoid(score)

            auroc, auprc, f1, mcc = calc_metrics(labels, score, classification=True)
            print(f"Run {i} AUROC: {auroc}, AUPRC: {auprc}, F1: {f1}, MCC: {mcc}")
            auroc_list.append(auroc)
            auprc_list.append(auprc)
            f1_list.append(f1)
            mcc_list.append(mcc)
            
        else:
            corr, rmse = calc_metrics(labels, score, classification=False)
            print(f"Run {i} Correlation: {corr}, RMSE: {rmse}")
            correlation_list.append(corr)
            rmse_list.append(rmse)
            
    if args.classification_task:
        for metric, metric_list in zip(["AUROC", "AUPRC", "F1", "MCC"], [auroc_list, auprc_list, f1_list, mcc_list]):
            print(f"{metric} mean: {np.mean(metric_list)}, std: {np.std(metric_list)}")
    else:
        for metric, metric_list in zip(["Correlation", "RMSE"], [correlation_list, rmse_list]):
            print(f"{metric} mean: {np.mean(metric_list)}, std: {np.std(metric_list)}")


# Benchmarking

# if __name__ == "__main__":
#     class BatchProfilerCallback(Callback):
#         def __init__(self, max_batches=1000):
#             super().__init__()
#             self.batch_times = []
#             self.batch_memory = []
#             self.start_time = None
#             self.max_batches = max_batches
            
#         def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#             if len(self.batch_times) >= self.max_batches:
#                 trainer.should_stop = True
#                 return
            
#             torch.cuda.reset_peak_memory_stats()
#             self.start_time = time.time()
            
#         def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#             if len(self.batch_times) >= self.max_batches:
#                 return
                
#             batch_time = time.time() - self.start_time
#             max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
#             self.batch_times.append(batch_time)
#             self.batch_memory.append(max_memory)
            
#             if len(self.batch_times) >= self.max_batches:
#                 trainer.should_stop = True
            
#         def get_stats(self):
#             time_mean = statistics.mean(self.batch_times)
#             time_std = statistics.stdev(self.batch_times) if len(self.batch_times) > 1 else 0
            
#             memory_mean = statistics.mean(self.batch_memory)
#             memory_std = statistics.stdev(self.batch_memory) if len(self.batch_memory) > 1 else 0
            
#             return {
#                 'batch_time_mean': time_mean,
#                 'batch_time_std': time_std,
#                 'batch_memory_mean': memory_mean,
#                 'batch_memory_std': memory_std,
#                 'total_batches': len(self.batch_times)
#             }

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_directory", type=str, required=True)
#     parser.add_argument("--embedding_dict_file_name", type=str, required=True)
#     parser.add_argument("--protein_embedding_dimension", type=int, required=True)
#     parser.add_argument("--module_type", type=str, required=True)
#     parser.add_argument("--learning_rate", type=float, required=True)
#     parser.add_argument(
#         "--classification_task",
#         type=lambda x: (str(x).lower() == "true"),
#         required=True,
#     )
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--num_batches", type=int, default=1000)

#     args = parser.parse_args()

#     train_df = pd.read_pickle(os.path.join(args.data_directory, "train.pkl"))
#     val_df = pd.read_pickle(os.path.join(args.data_directory, "val.pkl"))
#     test_df = pd.read_pickle(os.path.join(args.data_directory, "test.pkl"))


#     with open(os.path.join(args.data_directory, args.embedding_dict_file_name), "rb") as f:
#         protein_embeddings = pickle.load(f)

#     if args.module_type in {"attention_aggregation", "multihead_attention_aggregation", "swe"}:
#         collate_fn = attention_collate_fn
#     else:
#         collate_fn = non_attention_collate_fn

#     mean_agg, cls = False, False
#     if args.module_type == "mean_aggregation":
#         mean_agg = True
#     if args.module_type == "cls":
#         cls = True

#     torch.manual_seed(0)
#     datamodule = DTIDataModule(
#         protein_embeddings,
#         train_df,
#         val_df,
#         test_df,
#         collate_fn,
#         num_workers=8,
#         batch_size=args.batch_size,
#         mean_agg=mean_agg,
#         cls=cls,
#     )

#     num_steps = (
#         len(datamodule.train_dataset.drug_target_interactions) // args.batch_size
#     )
#     train_loader = datamodule.train_dataloader()
#     val_loader = datamodule.val_dataloader()

#     model = ModelModule(
#         num_steps,
#         module_type=args.module_type,
#         learning_rate=args.learning_rate,
#         protein_embedding_dim=args.protein_embedding_dimension,
#         attention_model_dim=args.protein_embedding_dimension,
#         classification=args.classification_task,
#     )
#     batch_profiler = BatchProfilerCallback(max_batches=args.num_batches)

#     trainer = L.Trainer(
#         max_epochs=-1,
#         devices=1,
#         accelerator="gpu",
#         log_every_n_steps=100,
#         precision="bf16-mixed",
#         callbacks=[batch_profiler],
#         enable_checkpointing=False,
#     )

#     trainer.fit(model, train_loader, val_loader)

#     stats = batch_profiler.get_stats()
#     print("\nbatch Processing Statistics:")

#     print(f"mean batch processing time: {stats['batch_time_mean']} ± {stats['batch_time_std']} seconds")
#     print(f"mean batch VRAM usage: {stats['batch_memory_mean']} ± {stats['batch_memory_std']} MB")


