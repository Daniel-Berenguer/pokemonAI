import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

cuda = torch.device('cuda')

CHART = [
# NONE  NOR  FIR  WAT  ELE  GRS  ICE  FIG  POI  GRO  FLY  PSY  BUG  ROC  GHO  DRG  DAR  STE  FAI
  [ 1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],   # None
  [ 1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 0.5,   0,   1,   1, 0.5,   1],  # Normal
  [ 1,  1, 0.5, 0.5,   1,   2,   2,   1,   1,   1,   1,   1,   2, 0.5,   1, 0.5,   1,   2,   1],  # Fire
  [ 1,  1,   2, 0.5,   1, 0.5,   1,   1,   1,   2,   1,   1,   1,   2,   1, 0.5,   1,   1,   1],  # Water
  [ 1,  1,   1,   2, 0.5, 0.5,   1,   1,   1,   0,   2,   1,   1,   1,   1, 0.5,   1,   1,   1],  # Electric
  [ 1,  1, 0.5,   2,   1, 0.5,   1,   1, 0.5,   2, 0.5,   1, 0.5,   2,   1, 0.5,   1, 0.5,   1],  # Grass
  [ 1,  1, 0.5, 0.5,   1,   2, 0.5,   1,   1,   2,   2,   1,   1,   1,   1,   2,   1, 0.5,   1],  # Ice
  [ 1,  2,   1,   1,   1,   1,   2,   1, 0.5,   1, 0.5, 0.5, 0.5,   2,   0,   1,   2,   2, 0.5],  # Fighting
  [ 1,  1,   1,   1,   1,   2,   1,   1, 0.5, 0.5,   1,   1,   1, 0.5, 0.5,   1,   1,   0,   2],  # Poison
  [ 1,  1,   2,   1,   2, 0.5,   1,   1,   2,   1,   0,   1, 0.5,   2,   1,   1,   1,   2,   1],  # Ground
  [ 1,  1,   1,   1, 0.5,   2,   2,   2,   1,   1,   1,   1,   2, 0.5,   1,   1,   1, 0.5,   1],  # Flying
  [ 1,  1,   1,   1,   1,   1,   1,   2,   2,   1,   1, 0.5,   1,   1,   1,   1,   0, 0.5,   1],  # Psychic
  [ 1,  1, 0.5,   1,   1,   2,   1, 0.5, 0.5,   1, 0.5,   2,   1,   1, 0.5,   1,   2, 0.5, 0.5],  # Bug
  [ 1,  1,   2,   1,   1,   1,   2, 0.5,   1, 0.5,   2,   1,   2,   1,   1,   1,   1, 0.5,   1],  # Rock
  [ 1,  0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   2,   1, 0.5,   1,   1],  # Ghost
  [ 1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1, 0.5,   0],  # Dragon
  [ 1,  1,   1,   1,   1,   1,   1, 0.5,   1,   1,   1,   2,   1,   1,   2,   1, 0.5,   1, 0.5],  # Dark
  [ 1,  1, 0.5, 0.5, 0.5,   1,   2,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1, 0.5,   2],  # Steel
  [ 1,  1, 0.5,   1,   1,   1,   1,   2, 0.5,   1,   1,   1,   1,   1,   1,   2,   2, 0.5,   1],  # Fairy
]

typeChart = torch.tensor(CHART)
print(typeChart[:,1])

    

class MoveEncoder(nn.Module):
    NUM_MOVES = 497
    MOVE_FEATS_SIZE = 5


    def __init__(self, nhidden, nTypes, move_dim):
        super().__init__()
        self.nTypes = nTypes

        # Receives moveInts (BATCH, TEAM, POKE, MOVES, moveIntDim)
        #          moveFeats (BATCH, TEAM, POKE, MOVES, moveFeatDim)

        self.emb = nn.Embedding(self.NUM_MOVES, move_dim) # Embeds move (move name, to capture effects I have not included in features)
        
        concat_dim = move_dim + nTypes + self.MOVE_FEATS_SIZE

        self.ln1 = nn.LayerNorm(concat_dim)
        self.lin1 = nn.Linear(concat_dim, nhidden)

        self.ln2 = nn.LayerNorm(nhidden)
        self.lin2 = nn.Linear(nhidden, move_dim)

    def forward(self, moveInts, moveFeats):
        move_name = moveInts[:, :, :, :, 0]
        move_type = moveInts[:, :, :, :, 1]

        concat = torch.cat(
            (self.emb(move_name), F.one_hot(move_type, num_classes=self.nTypes), moveFeats)
        ) # (BATCH, TEAM, POKE, MOVES, move_dim + type_dim + moveFeatDim)

        x = torch.relu(self.lin1(self.ln1(concat))) # (BATCH, TEAM, POKE, MOVES, nHidden)

        # Sum across move dim
        x = x.sum(dim=3) # (BATCH, TEAM, POKE, nHidden)

        x = torch.relu(self.lin2(self.ln2(x))) # (BATCH, TEAM, POKE, moveDim)
        
        return x


class AttentionHead(nn.Module):
    def __init__(self, in_dim, key_query_dim, value_dim):
        super().__init__()
        self.key_query_dim = key_query_dim
        self.Wq = nn.Linear(in_dim, key_query_dim, bias=False)
        self.Wk = nn.Linear(in_dim, key_query_dim, bias=False)
        self.Wv = nn.Linear(in_dim, value_dim, bias=False)

    def forward(self, q_in, k_in, v_in):
        # (BATCH, ???, featDim)
        Q = self.Wq(q_in) # (BATCH, 2, 6, KQ_dim)
        K = self.Wk(k_in) # (BATCH, 2, 6, KQ_dim)
        V = self.Wv(v_in) # (BATCH, 2, 6, V_dim)
        Y = torch.softmax(Q @ K.transpose(-2,-1) / torch.sqrt(self.key_query_dim), dim=-1) @ V
        return Y
    

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, n_heads, key_query_dim, head_dim):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(in_dim, key_query_dim, head_dim) for _ in range(n_heads)])
        self.Wo = nn.Linear(n_heads * head_dim, in_dim)

    def forward(self, q_in, k_in, v_in):
        out = torch.cat([h(q_in, k_in, v_in) for h in self.heads], dim=-1)
        return self.Wo(out)


class SelfAttTransformerLayer(nn.Module):
    def __init__(self, in_dim, n_heads, key_query_dim, head_dim, hidden_dim, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(in_dim, n_heads, key_query_dim, head_dim)
        self.ln1 = nn.LayerNorm(in_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.ln2 = nn.LayerNorm(in_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        normed_x = self.ln1(x)
        x = x + self.dropout1(self.mha(normed_x, normed_x, normed_x))
        x = x + self.dropout2(self.lin2(self.relu(self.lin1(self.ln2(x)))))
        return x

class PokeEncoder(nn.Module):
    N_POKES = 228
    N_ABS = 200
    N_ITEMS = 149
    FEATS_DIM = 24
    N_TYPES = 19

    TYPE_CHART = [
    # NONE  NOR  FIR  WAT  ELE  GRS  ICE  FIG  POI  GRO  FLY  PSY  BUG  ROC  GHO  DRG  DAR  STE  FAI
    [ 1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1],   # None
    [ 1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1, 0.5,   0,   1,   1, 0.5,   1],  # Normal
    [ 1,  1, 0.5, 0.5,   1,   2,   2,   1,   1,   1,   1,   1,   2, 0.5,   1, 0.5,   1,   2,   1],  # Fire
    [ 1,  1,   2, 0.5,   1, 0.5,   1,   1,   1,   2,   1,   1,   1,   2,   1, 0.5,   1,   1,   1],  # Water
    [ 1,  1,   1,   2, 0.5, 0.5,   1,   1,   1,   0,   2,   1,   1,   1,   1, 0.5,   1,   1,   1],  # Electric
    [ 1,  1, 0.5,   2,   1, 0.5,   1,   1, 0.5,   2, 0.5,   1, 0.5,   2,   1, 0.5,   1, 0.5,   1],  # Grass
    [ 1,  1, 0.5, 0.5,   1,   2, 0.5,   1,   1,   2,   2,   1,   1,   1,   1,   2,   1, 0.5,   1],  # Ice
    [ 1,  2,   1,   1,   1,   1,   2,   1, 0.5,   1, 0.5, 0.5, 0.5,   2,   0,   1,   2,   2, 0.5],  # Fighting
    [ 1,  1,   1,   1,   1,   2,   1,   1, 0.5, 0.5,   1,   1,   1, 0.5, 0.5,   1,   1,   0,   2],  # Poison
    [ 1,  1,   2,   1,   2, 0.5,   1,   1,   2,   1,   0,   1, 0.5,   2,   1,   1,   1,   2,   1],  # Ground
    [ 1,  1,   1,   1, 0.5,   2,   2,   2,   1,   1,   1,   1,   2, 0.5,   1,   1,   1, 0.5,   1],  # Flying
    [ 1,  1,   1,   1,   1,   1,   1,   2,   2,   1,   1, 0.5,   1,   1,   1,   1,   0, 0.5,   1],  # Psychic
    [ 1,  1, 0.5,   1,   1,   2,   1, 0.5, 0.5,   1, 0.5,   2,   1,   1, 0.5,   1,   2, 0.5, 0.5],  # Bug
    [ 1,  1,   2,   1,   1,   1,   2, 0.5,   1, 0.5,   2,   1,   2,   1,   1,   1,   1, 0.5,   1],  # Rock
    [ 1,  0,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1,   1,   2,   1, 0.5,   1,   1],  # Ghost
    [ 1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   1, 0.5,   0],  # Dragon
    [ 1,  1,   1,   1,   1,   1,   1, 0.5,   1,   1,   1,   2,   1,   1,   2,   1, 0.5,   1, 0.5],  # Dark
    [ 1,  1, 0.5, 0.5, 0.5,   1,   2,   1,   1,   1,   1,   1,   1,   2,   1,   1,   1, 0.5,   2],  # Steel
    [ 1,  1, 0.5,   1,   1,   1,   1,   2, 0.5,   1,   1,   1,   1,   1,   1,   2,   2, 0.5,   1],  # Fairy
    ]

    def __init__(self, nhidden, moveHidden, moveDim, typeEmb, pokeEmb, abEmb, itemEmb, n_heads):
        super().__init__()

        self.pokeEmb = nn.Embedding(self.N_POKES, pokeEmb)
        self.abEmb = nn.Embedding(self.N_ABS, abEmb)
        self.itemEmb = nn.Embedding(self.N_ITEMS, itemEmb)
        self.type_chart = torch.tensor(self.TYPE_CHART, requires_grad=False)

        self.moveEncoder = MoveEncoder(nhidden=moveHidden, move_dim=moveDim, nTypes=self.N_TYPES)

        """self.typeW = nn.Parameter(torch.randn(typeEmb, nhidden))
        nn.init.kaiming_normal_(self.typeW)
        self.teraW = nn.Parameter(torch.randn(typeEmb, nhidden))
        nn.init.kaiming_normal_(self.teraW)
        self.featsW = nn.Parameter(torch.randn(self.FEATS_DIM, nhidden))
        nn.init.kaiming_normal_(self.featsW)"""

        concatDim = moveDim + pokeEmb + abEmb + itemEmb + self.N_TYPES*4 + self.FEATS_DIM
        self.selfAttTransformerLayer = SelfAttTransformerLayer(concatDim, 10, concatDim, concatDim/n_heads, concatDim*2, dropout=0.3)

    def forward(self, pokeInts, pokeFeats, moveInts, moveFeats):
        # Encodes Moves
        moves = self.moveEncoder.forward(moveInts, moveFeats, self.typeEncoder)
        
        # Embeds Rest
        # pokeInts (BATCH, 2, 6, 6) [poke, item, ab, typ1, typ2]
        poke = self.pokeEmb(pokeInts[:, :, :, 0])
        item = self.itemEmb(pokeInts[:, :, :, 1])
        ab = self.abEmb(pokeInts[:, :, :, 2])
        types = F.one_hot(pokeInts[:, :, :, 3:5], num_classes=self.N_TYPES).flatten(start_dim=3)
        types_def = self.type_chart[:, pokeInts[:, :, :, 3:5]].flatten(start_dim=3)

        # Concatenate everything
        concat = torch.cat([poke, item, ab, types, pokeFeats, types_def, moves], dim=-1) # (BATCH, 2, 6, concatDim)
        x = self.selfAttTransformerLayer(concat)

        
        return torch.relu(self.linear(self.ln(concat)))
    

class BoardEncoder(nn.Module):
    NTURNS = 5
    NWEATHERS = 5
    NTERRAINS = 5
    NEMB = 4
    NFEATS = 15
    
    def __init__(self, nhidden):
        super().__init__()
        self.twEmb = nn.Embedding(self.NTURNS, self.NEMB)
        self.trEmb = nn.Embedding(self.NTURNS, self.NEMB)
        self.weatherEmb = nn.Embedding(self.NWEATHERS, self.NEMB)
        self.terrainEmb = nn.Embedding(self.NTERRAINS, self.NEMB)

        newDim = self.NEMB*3 + self.NEMB + self.NEMB + self.NFEATS

        self.ln = nn.LayerNorm(newDim)
        self.lin = nn.Linear(newDim, nhidden)

    def forward(self, boardInts, boardFeats):
        # Embeds Ints 
        tws = torch.flatten(self.twEmb(boardInts[:, 0:2]), start_dim=1)
        tr = self.trEmb(boardInts[:, 2])
        weather = self.weatherEmb(boardInts[:, 3])
        terrain = self.terrainEmb(boardInts[:, 4])

        # Concats
        comb = torch.cat([tws, tr, weather, terrain, boardFeats], dim=-1)

        return torch.relu(self.lin(self.ln(comb)))