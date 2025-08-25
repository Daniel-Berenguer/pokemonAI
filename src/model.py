import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

cuda = torch.device('cuda')
    
#boardIntTensor (5) [tw1, tw2, tr, weather, terrain]
#boardTensor (boardFDim=15)
#pokeIntsTensor (2, 6, 6) [poke, item, ab, typ1, typ2, tera]
#pokeFeatsTensor (2, 6, pokeFeatDim=17)
#moveIntsTensor (2, 6, 4, 2) [moveName, moveType]
#moveFeatsTensor (2, 6, 4, moveFeatDim)


class TypeEncoder(nn.Module):
    NUM_TYPES = 20
    def __init__(self, nhidden=10):
        super().__init__()
        self.emb = nn.Embedding(self.NUM_TYPES, nhidden)

    def forward(self, x):
        return self.emb(x)
    

class MoveEncoder(nn.Module):
    NUM_MOVES = 382
    MOVE_FEATS_SIZE = 6
    MOVE_DIM = 64
    def __init__(self, nhidden=256, typeEmbeddingSize=10):
        super().__init__()
        self.emb = nn.Embedding(self.NUM_MOVES, self.MOVE_DIM) # Embeds move (move name, to capture effects I have not included in features)
        self.U = nn.Parameter(torch.randn(typeEmbeddingSize, self.MOVE_DIM)) # Multiplies type embedding before adding to move embedding
        self.W = nn.Parameter(torch.randn(self.MOVE_FEATS_SIZE, self.MOVE_DIM)) # Multiplies move features (e.g power) before adding to move embedding
        self.lin = nn.Linear(self.MOVE_DIM, nhidden)

    def forward(self, moveInts, moveFeats, typeEncoder : TypeEncoder):
        move_name = moveInts[:, :, :, :, 0]
        move_type = moveInts[:, :, :, :, 1]
        comb = torch.relu(self.emb(move_name) + typeEncoder.forward(move_type) @ self.U + moveFeats @ self.W)
        return torch.relu(self.lin(comb)).sum(dim=3)
    
class PokeEncoder(nn.Module):
    N_POKES = 261
    N_ABS = 212
    N_ITEMS = 133
    FEATS_DIM = 17

    POKE_EMB = 64
    AB_EMB = 128
    ITEM_EMB = 64

    def __init__(self, nhidden=512, moveHidden=256, typeEmb=10):
        super().__init__()

        self.pokeEmb = nn.Embedding(self.N_POKES, self.POKE_EMB)
        self.abEmb = nn.Embedding(self.N_ABS, self.AB_EMB)
        self.itemEmb = nn.Embedding(self.N_ITEMS, self.ITEM_EMB)

        self.moveEncoder = MoveEncoder(nhidden=moveHidden, typeEmbeddingSize=typeEmb)
        self.typeEncoder = TypeEncoder(nhidden=typeEmb)

        """self.typeW = nn.Parameter(torch.randn(typeEmb, nhidden))
        nn.init.kaiming_normal_(self.typeW)
        self.teraW = nn.Parameter(torch.randn(typeEmb, nhidden))
        nn.init.kaiming_normal_(self.teraW)
        self.featsW = nn.Parameter(torch.randn(self.FEATS_DIM, nhidden))
        nn.init.kaiming_normal_(self.featsW)"""

        concatDim = moveHidden + self.POKE_EMB + self.ITEM_EMB + self.AB_EMB + typeEmb*3 + self.FEATS_DIM
        self.linear = nn.Linear(concatDim, nhidden)

    def forward(self, pokeInts, pokeFeats, moveInts, moveFeats):
        # Encodes Moves
        moves = self.moveEncoder.forward(moveInts, moveFeats, self.typeEncoder)
        
        # Embeds Rest
        # pokeInts (BATCH, 2, 6, 6) [poke, item, ab, typ1, typ2, tera]
        poke = self.pokeEmb(pokeInts[:, :, :, 0])
        item = self.itemEmb(pokeInts[:, :, :, 1])
        ab = self.abEmb(pokeInts[:, :, :, 2])
        types = self.typeEncoder(pokeInts[:, :, :, 3:6]).flatten(start_dim=3)

        # Concatenate everything
        concat = torch.cat([poke, item, ab, types, pokeFeats, moves], dim=-1)
        
        return torch.relu(self.linear(concat))

    
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

        self.bn = nn.BatchNorm1d(newDim)
        self.lin = nn.Linear(newDim, nhidden)

    def forward(self, boardInts, boardFeats):
        # Embeds Ints 
        tws = torch.flatten(self.twEmb(boardInts[:, 0:2]), start_dim=1)
        tr = self.trEmb(boardInts[:, 2])
        weather = self.weatherEmb(boardInts[:, 3])
        terrain = self.terrainEmb(boardInts[:, 4])

        # Concats
        comb = torch.cat([tws, tr, weather, terrain, boardFeats], dim=-1)

        return torch.relu(self.lin(self.bn(comb)))

#boardIntTensor (5) [tw1, tw2, tr, weather, terrain]
#boardTensor (boardFDim=15)
#pokeIntsTensor (2, 6, 6) [poke, item, ab, typ1, typ2, tera]
#pokeFeatsTensor (2, 6, pokeFeatDim=17)
#moveIntsTensor (2, 6, 4, 2) [moveName, moveType]
#moveFeatsTensor (2, 6, 4, moveFeatDim)
class Model(nn.Module):
    BOARD_HIDDEN = 64
    TYPE_EMB = 10
    POKE_HIDDEN = 512
    MOVE_HIDDEN = 256
    DROPOUT_PROB = 0.3

    def __init__(self):
        super().__init__()
        self.boardEncoder = BoardEncoder(nhidden=self.BOARD_HIDDEN)
        self.pokeEncoder = PokeEncoder(nhidden=self.POKE_HIDDEN, moveHidden=self.MOVE_HIDDEN, typeEmb=self.TYPE_EMB)

        self.pokeW = nn.Parameter(torch.randn(self.POKE_HIDDEN, self.POKE_HIDDEN*4))
        nn.init.kaiming_normal_(self.pokeW)

        concatDim = self.POKE_HIDDEN * 8 + self.BOARD_HIDDEN

        self.linearLayers = nn.ModuleList([nn.Linear(concatDim, 512), nn.Linear(512, 256), nn.Linear(256, 64), nn.Linear(64, 16)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(concatDim), nn.BatchNorm1d(512), nn.BatchNorm1d(256), nn.BatchNorm1d(64)])
        
        # Add Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=self.DROPOUT_PROB),
            nn.Dropout(p=self.DROPOUT_PROB),
            nn.Dropout(p=self.DROPOUT_PROB),
            nn.Dropout(p=self.DROPOUT_PROB)
        ])
                                 
        self.finalLinear = nn.Linear(16, 1)

    def forward(self, boardInts, boardFeats, pokeInts, pokeFeats, moveInts, moveFeats):
        board = self.boardEncoder.forward(boardInts, boardFeats)
        pokes = self.pokeEncoder.forward(pokeInts, pokeFeats, moveInts, moveFeats)
        
        # pokes shape is (BATCH, 2, 6, pokeDIM)
        teams = (pokes @ self.pokeW).sum(dim=2).flatten(start_dim=1)
        x = torch.cat([board, teams], dim=-1)
        # x shape is (BATCH, pokeDIM*4*2 + boardDIm)

        for bn,linear,drop in zip(self.bns,self.linearLayers,self.dropouts):
            x = torch.relu(linear(bn(x)))
            x = drop(x)


        return self.finalLinear(x).reshape(-1)


if __name__ == "__main__":

    with open("data/data.pickle", "rb") as file:
        # X is boardIntTensor, boardTensor, pokeIntsTensor, pokeFeatsTensor, moveFeatsTensor, moveIntsTensor
        X, Y = pickle.load(file)


    model = Model().to(cuda)
    nEl = 0
    for p in model.parameters():
        nEl += p.numel()
    print(f"Number of params: {nEl}")


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


    # Shuffle
    indices = torch.randperm(X[0].size(0))  # random permutation of indices
    for i, tens in enumerate(X):
        X[i] = tens[indices]
    Y = Y[indices]

    n = int(Y.size(0)*0.85)
    print(f"Dataset size: {Y.size(0)}")

    X_train = [X[i][:n] for i in range(len(X))]
    Y_train = Y[:n]
    X_test =  [X[i][n:] for i in range(len(X))]
    Y_test = Y[n:]


    EPOCHS = 20
    BATCH_SIZE = 16
    ITERS = int((n * EPOCHS)/BATCH_SIZE)
    CHECK_INTERVAL = 100
    

    loss_avg = 0

    for i in range(ITERS+1):
        ix = torch.randint(0, Y_train.size(0), (BATCH_SIZE,))  # random indices
        #boardIntTensor, boardTensor, pokeIntsTensor, pokeFeatsTensor, moveFeatsTensor, moveIntsTensor
        boardIntBatch = X_train[0][ix].to(cuda)
        boardFeatBatch = X_train[1][ix].to(cuda)
        pokeIntBatch = X_train[2][ix].to(cuda)
        pokeFeatBatch = X_train[3][ix].to(cuda)
        moveIntBatch = X_train[4][ix].to(cuda)
        moveFeatBatch = X_train[5][ix].to(cuda)

        Y_batch = Y_train[ix].to(cuda)

        logits = model.forward(boardIntBatch, boardFeatBatch, pokeIntBatch, pokeFeatBatch, moveIntBatch, moveFeatBatch)

        loss = F.binary_cross_entropy_with_logits(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg += loss.item()
        if i % CHECK_INTERVAL == 0 or i == ITERS:
            if i != 0:
                loss_avg /= CHECK_INTERVAL
            print(f'Iteration [{i}/{ITERS}], Loss: {loss_avg:.4f}')
            loss_avg = 0


    # Test loss
    with torch.no_grad():
        boardIntTest = X_test[0].to(cuda)
        boardFeatTest = X_test[1].to(cuda)
        pokeIntTest = X_test[2].to(cuda)
        pokeFeatTest = X_test[3].to(cuda)
        moveIntTest = X_test[4].to(cuda)
        moveFeatTest = X_test[5].to(cuda)
        Y_test = Y_test.to(cuda)
        logits = model.forward(boardIntTest, boardFeatTest, pokeIntTest, pokeFeatTest, moveIntTest, moveFeatTest)
        loss = F.binary_cross_entropy_with_logits(logits, Y_test)
        print(f"Test loss: {loss.item():.4f}")
        probs = F.sigmoid(logits)
        predicts = torch.round(probs)
        correct = torch.eq(predicts, Y_test).sum()
        acc = correct / Y_test.size(0)
        print(f"Test accuracy: {acc.item():.4f}")

    with open("data/model_state_dict", "wb") as file:
        torch.save(model.state_dict(), file)