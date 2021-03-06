from time import sleep
from copy import deepcopy
from pathlib import Path

#from tqdm import tqdm
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from kif import Kif


class Dataset(torch.utils.data.Dataset):
    def __init__(self, kif_files):
        """
        Args:
            kif_files (list[str]): 棋譜ファイル名のリスト
        """

        self.kif_files = kif_files
        self.kifs = self.load_all_kif_files(kif_files)

    def load_all_kif_files(self, kif_files):
        kifs = []  # type: List[Kif]
        for file in kif_files:
            kif = Kif.from_file(file)
            kifs.append(kif)
        return kifs

    def __getitem__(self, idx):
        """
        Args:
            idx (int):
        Returns:
            torch.Tensor: agent_features     エージェント特徴 (4, length) dtype=long
            torch.Tensor: condition_features 状態特徴        (length,)   dtype=long
            torch.Tensor: target_rank        最終順位        (4,)        dtype=long
            torch.Tensor: target_policy      探索で得た方策   (4, 4)      dtype=float
        """

        kif = self.kifs[idx]
        n_steps = len(kif.steps) - 1  # 最後のステップは着手が無いので除外
        step = torch.randint(n_steps, tuple())
        agent_features = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(feats) for feats in kif.steps[step].agent_features],
            batch_first=True, padding_value=-100
        ).to(torch.long)  # padding_value を設定すると float になるので戻す必要がある
        condition_features = torch.tensor(kif.steps[step].condition_features)
        target_rank = torch.tensor(kif.ranks)
        target_policy = torch.tensor(kif.steps[step].values, dtype=torch.float)[:, 1:]
        assert target_policy.shape == (4, 4)
        return agent_features, condition_features, target_rank, target_policy

    def __len__(self):
        return len(self.kif_files)


class FastDataset(torch.utils.data.Dataset):
    def __init__(self, kif_files):
        """局面を直接持っておくデータセット
        
        Args:
            kif_files (list[str]): 棋譜ファイル名のリスト
        """

        self.kif_files = kif_files
        self.data = self.load_all_states(kif_files)

    def load_all_states(self, kif_files):
        data = []  # type: 
        for file in tqdm(kif_files):
            kif = Kif.from_file(file)
            if kif is None:
                continue
            for step in kif.steps[:-1]:  # 最後のステップは着手が無いので除外
                agent_features = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(feats) for feats in step.agent_features],
                    batch_first=True, padding_value=-100
                ).to(torch.long)  # padding_value を設定すると float になるので戻す必要がある
                condition_features = torch.tensor(step.condition_features)
                target_rank = torch.tensor(kif.ranks)
                target_policy = torch.tensor(step.values, dtype=torch.float)[:, 1:]
                assert target_policy.shape == (4, 4)
                data.append((agent_features, condition_features, target_rank, target_policy))
        return data

    def __getitem__(self, idx):
        """
        Args:
            idx (int):
        Returns:
            torch.Tensor: agent_features     エージェント特徴 (4, length) dtype=long
            torch.Tensor: condition_features 状態特徴        (length,)   dtype=long
            torch.Tensor: target_rank        最終順位        (4,)        dtype=long
            torch.Tensor: target_policy      探索で得た方策   (4, 4)      dtype=float
        """

        return self.data[idx]

    def __len__(self):
        return len(self.data)

    
class EfficientDataset(torch.utils.data.Dataset):
    def __init__(self, kif_files, max_data_size=200*100000):
        """局面をテンソルにまとめて持っておくデータセット
        
        Args:
            kif_files (list[str]): 棋譜ファイル名のリスト
        """

        self.kif_files = kif_files
        self.max_data_size = max_data_size  # -> 2e7
        self.max_feature_size = 50
        
        self.agent_feature_buffer     = torch.full((self.max_data_size, 4, self.max_feature_size), -100, dtype=torch.int16)
        self.condition_feature_buffer = torch.full((self.max_data_size, 3), -100, dtype=torch.int16)
        self.target_rank_buffer       = torch.full((self.max_data_size, 4), -100, dtype=torch.int8)
        self.target_policy_buffer     = torch.full((self.max_data_size, 4, 4), -100, dtype=torch.float32)
        
        self.data_size = 0
        self.load_all_states(kif_files)
    
    def copy(self):
        new_dataset = EfficientDataset([])
        new_dataset.agent_feature_buffer[:self.data_size]     = self.agent_feature_buffer[:self.data_size]
        new_dataset.condition_feature_buffer[:self.data_size] = self.condition_feature_buffer[:self.data_size]
        new_dataset.target_rank_buffer[:self.data_size]       = self.target_rank_buffer[:self.data_size]
        new_dataset.target_policy_buffer[:self.data_size]     = self.target_policy_buffer[:self.data_size]
        new_dataset.data_size = self.data_size
        return new_dataset
    
    def load_all_states(self, kif_files):
        idx = self.data_size
        for file in tqdm(kif_files):
            kif = Kif.from_file(file)
            if kif is None:
                continue
            for step in kif.steps[:-1]:  # 最後のステップは着手が無いので除外
                agent_features = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(feats) for feats in step.agent_features],
                    batch_first=True, padding_value=-100
                ).to(torch.int16)  # padding_value を設定すると float になるので戻す必要がある
                condition_features = torch.tensor(step.condition_features)
                target_rank = torch.tensor(kif.ranks)
                target_policy = torch.tensor(step.values, dtype=torch.float)[:, 1:]
                assert target_policy.shape == (4, 4)
                
                self.agent_feature_buffer[idx, :, :agent_features.shape[1]] = agent_features
                self.condition_feature_buffer[idx] = condition_features
                self.target_rank_buffer[idx] = target_rank
                self.target_policy_buffer[idx] = target_policy
                
                idx += 1
        
        self.data_size = idx

    def __getitem__(self, idx):
        """
        これは嘘
        
        Args:
            idx (int):
        Returns:
            torch.Tensor: agent_features     エージェント特徴 (4, length) dtype=long
            torch.Tensor: condition_features 状態特徴        (length,)   dtype=long
            torch.Tensor: target_rank        最終順位        (4,)        dtype=long
            torch.Tensor: target_policy      探索で得た方策   (4, 4)      dtype=float
        """

        return idx, self

    def __len__(self):
        return self.data_size


class Model(nn.Module):
    def __init__(self, features, out_dim=5, hidden_1=256, hidden_2=32):
        super().__init__()
        self.features = features
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.embed = nn.EmbeddingBag(features+1, hidden_1, mode="sum", padding_idx=features)
        self.embed.weight.data /= 16.0  # 小さめの値で初期化
        self.linear_condition = nn.Linear(hidden_1, hidden_2)
        self.linear_condition.weight.data #/= 2.0
        self.linear_condition.bias.data += 0.5
        self.linear_2 = nn.Linear(hidden_1, hidden_2)
        self.linear_2.weight.data /= 2.0
        self.linear_2.bias.data += 0.5
        self.linear_3 = nn.Linear(hidden_2, hidden_2)
        self.linear_3.weight.data /= 2.0
        self.linear_3.bias.data += 0.5
        self.linear_4 = nn.Linear(hidden_2, out_dim)
        #self.linear_4.bias.data += 0.5
        self.quantized = False

    def forward(self, x, condition, ignore_rounding=False, debug=False):
        """

        Args:
            x: [[1, 3, 9, ..., 22, -100, -100], ... ] みたいなテンソル (batch, 4, length)
            condition: (batch, length)
        Returns:
            torch.Tensor: (batch, 4, out_dim)
        """

        def hardtanh_(x, limit, p):
            if self.quantized:
                return F.hardtanh(x, -limit, limit, inplace=True)
            else:
                return F.hardtanh(x, -limit / (1 << p), limit / (1 << p), inplace=False)

        def scale(x, p):
            if self.quantized:
                if ignore_rounding:
                    return x * 2 ** p
                else:
                    return torch.round(x * 2 ** p)
            else:
                return x

        def clipped_relu(x, limit, p):
            if self.quantized:
                return F.hardtanh(x, 0.0, limit, inplace=True)
            else:
                return F.hardtanh(x, 0.0, limit / (1 << p), inplace=False)

        def add(x, a, p):
            if self.quantized:
                return x + a * (1 << p)
            else:
                return x + a
        
        batch_size = x.shape[0]
        x = torch.where(x != -100, x, self.features)

        # (1) [batch, 4, length] -> [batch, 4, 256]
        x = self.embed(x.view(batch_size * 4, -1)).view(batch_size, 4, self.hidden_1)  # scale = 2^11, max = 2^15
        x = add(x, 0.5, 11)                                    # μ=0.5, σ=1.0, min=-15.5, max=16.5 | μ=2^10, σ=2^11, min=-...
        x = clipped_relu(x, 127 << 4, 11)                      # μ=0.5, σ=1.0, min=0, max=127/128 | μ=2^10, σ=2^11, min=0, max=127<<4      # scale = 2^11, max = (2^7-1)2^4
        
        if debug:
            print(f"1) {x[0, :3, :5] / ((1<<11) if self.quantized else 1)}")

        # (2) [batch, length] -> [batch, 256]
        condition = self.embed(condition)                      # μ=0, σ=1.0, min=-16, max=16 | μ=0, σ=2^11, min=-2^15, max=2-15
        condition = add(condition, 0.5, 11)                    # μ=0.5, σ=1.0, min=-15.5, max=16.5 | μ=2^10, σ=2^11, min=...
        condition = clipped_relu(condition, 127 << 4, 11)      # μ=0.5, σ=1.0, min=0, max=127/128 | μ=2^10, σ=2^11, min=0, max=127<<5        # scale = 2^11, max = 127<<4
        condition = condition + x.sum(1)/4                     # μ=1.0, σ=2.0, min=0, max=2*127/128 | μ=2^11, σ=2^12, min=0, max=127<<5      # scale = 2^11 max = 127<<5
        if not self.quantized:
            condition *= 0.5                                   # μ=0.5, σ=1.0, min=0, max=127/128 | ...
        condition = scale(condition, -5)                       # μ=0.5, σ=1.0, min=0, max=127/128 | μ=2^6, σ=2^7, min=0, max=127            # scale = 2^7,  max = 2^7-1

        if debug:
            print(f"2) {condition[0, :20] / ((1<<7) if self.quantized else 1)}")
        
        # (3) [batch, 256] -> [batch, 32]
        condition = self.linear_condition(condition)           # scale = 2^16
        condition = clipped_relu(condition, 127 << 9, 16)      # scale = 2^16, max = (2^7-1)2^9
        #condition = scale(condition, 1)                        # scale = 2^16, max = (2^7-1)2^9
        
        if debug:
            print(f"3) {condition[0, :3] / ((1<<16) if self.quantized else 1)}")
        
        # (4) [batch, 4, 256] -> [batch, 4, 32]
        x = scale(x, -4)                                       # scale = 2^7,  max = 2^7-1
        x = self.linear_2(x)                                   # scale = 2^16
        x = clipped_relu(x, 127 << 9, 16)                      # scale = 2^16, max = (2^7-1)2^9
        x = x + condition.unsqueeze(1)                         # scale = 2^16, max = (2^7-1)2^10
        if not self.quantized:
            condition *= 0.5
        x = scale(x, -10)                                       # scale = 2^6,  max = 2^7-1

        if debug:
            print(f"4) {x[0, :3] / ((1<<6) if self.quantized else 1)}")
        
        # (5) [batch, 4, 32] -> [batch, 4, 32]
        x = self.linear_3(x)                                   # scale = 2^14
        x = clipped_relu(x, 127 << 7, 14)                      # scale = 2^14, max = (2^7-1)2^7
        x = scale(x, -7)                                       # scale = 2^7,  max = 2^7-1

        if debug:
            print(f"5) {x[0, :3] / ((1<<7) if self.quantized else 1)}")
        
        # (6) [batch, 4, 32] -> [batch, 4, out_dim]
        x = self.linear_4(x)                                   # scale = 2^13
        if self.quantized:
            x /= 1 << 13  # float に変換                        # scale = 1

        return x * 2.0

    def quantize(self, ignore_rounding=False):
        qmodel = deepcopy(self)
        qmodel.quantized = True

        # scaling
        def scale(params, p, bits=8):
            device = params.data.device
            mi = torch.tensor(-(1<<bits-1), dtype=torch.float).to(device)
            ma = torch.tensor((1<<bits-1)-1, dtype=torch.float).to(device)
            params.data = params.data * (1 << p)
            if not ignore_rounding:
                params.data = torch.round(params.data)
            params.data = torch.where(params.data > ma, ma, params.data)
            params.data = torch.where(params.data < mi, mi, params.data)
        scale(qmodel.embed.weight, 11, bits=16)
        scale(qmodel.linear_condition.weight, 9)
        scale(qmodel.linear_condition.bias, 16, bits=32)
        scale(qmodel.linear_2.weight, 9)
        scale(qmodel.linear_2.bias, 16, bits=32)
        scale(qmodel.linear_3.weight, 8)
        scale(qmodel.linear_3.bias, 14, bits=32)
        scale(qmodel.linear_4.weight, 6)
        scale(qmodel.linear_4.bias, 13, bits=32)

        return qmodel
    
    def clip_params(self, soft=False):
        def clip(params, p, bits=8):
#             device = params.data.device
#             mi = torch.tensor(-(1<<bits-1), dtype=torch.float).to(device) / (1 << p)
#             ma = torch.tensor((1<<bits-1)-1, dtype=torch.float).to(device) / (1 << p)
#             params.data = torch.where(params.data > ma, ma, params.data)
#             params.data = torch.where(params.data < mi, mi, params.data)
            mi = -(1<<bits-1) / (1 << p)
            ma = ((1<<bits-1)-1) / (1 << p)
            if soft:
                params.data = torch.tanh(params.data / ma) * ma
            else:
                params.data = F.hardtanh_(params.data, mi, ma)
        if soft:
            clip(self.embed.weight, 11, bits=11)
        else:
            clip(self.embed.weight, 11, bits=12)
        clip(self.linear_condition.weight, 9)
        clip(self.linear_condition.bias, 16, bits=32)
        clip(self.linear_2.weight, 9)
        clip(self.linear_2.bias, 16, bits=32)
        clip(self.linear_3.weight, 8)
        clip(self.linear_3.bias, 14, bits=32)
        clip(self.linear_4.weight, 6)
        clip(self.linear_4.bias, 13, bits=32)
        
    def dump(self, filename):
        assert self.quantized
        with open(filename, "wb") as f:
            def write(params, dtype="int8"):
                f.write(params.detach().numpy().ravel().astype(dtype).tobytes())
            write(self.embed.weight, "int16")
            write(self.linear_condition.weight)
            write(self.linear_condition.bias, "int32")
            write(self.linear_2.weight)
            write(self.linear_2.bias, "int32")
            write(self.linear_3.weight)
            write(self.linear_3.bias, "int32")
            write(self.linear_4.weight)
            write(self.linear_4.bias, "int32")


def soft_cross_entropy(pred, target):
    """ソフトラベルのクロスエントロピー

    全バッチの合計

    Args:
        pred: softmax 前の値 (batch, n)
        target: (batch, n)

    Returns:
        torch.Tensor: クロスエントロピー (全バッチ合計)
    """
    return -(target*F.log_softmax(pred, dim=1)).sum()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target_rank, target_policy):
        """

        Args:
            x             (torch.Tensor): モデルの出力   (batch, 4, 5) dtype=float
            target_rank   (torch.Tensor): 最終順位      (batch, 4)    dtype=long
            target_policy (torch.Tensor): 探索で得た方策 (batch, 4, 4) dtype=float
        Returns:
            torch.Tensor: loss (平均)
        """
        batch_size = x.shape[0]
        device = x.device
        
        value_loss = 0.0
        for a in range(4):
            for b in range(a+1, 4):
                rank_diff = target_rank[:, a] - target_rank[:, b]  # (batch,)
                t = torch.where(rank_diff < 0, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))
                t = torch.where(rank_diff == 0, torch.tensor(0.5).to(device), t)
                pred = x[:, a, 0] - x[:, b, 0]
                value_loss = value_loss + F.binary_cross_entropy_with_logits(pred, t)
        value_loss /= 6.0  # 4C2

        pred = x[:, :, 1:].reshape(batch_size * 4, 4)
        alive = target_policy[:, :, 0] != -100.0  # (batch, 4)
        target = (target_policy*alive.unsqueeze(2)).view(batch_size*4, 4)
        policy_loss = soft_cross_entropy(pred, target) * 0.5 / (alive.sum()+1e-10)

        #print("loss:", value_loss.item(), policy_loss.item())
        loss = value_loss + policy_loss
        return loss


# def collate_fn(batch):
#     agent_features, condition_features, target_rank, target_policy = zip(*batch)
#     pad_sequence = torch.nn.utils.rnn.pad_sequence
#     agent_features = pad_sequence([f.permute(1, 0) for f in agent_features], batch_first=True,
#                                   padding_value=-100).permute(0, 2, 1).contiguous()
#     condition_features = pad_sequence(condition_features, batch_first=True, padding_value=-100).contiguous()
#     target_rank = torch.stack(target_rank)
#     target_policy = torch.stack(target_policy)
#     return agent_features, condition_features, target_rank, target_policy

def collate_fn(batch):
    idxs, (dataset, *_) = zip(*batch)
    idxs = list(idxs)
    return dataset.agent_feature_buffer[idxs].to(torch.long), dataset.condition_feature_buffer[idxs].to(torch.long), dataset.target_rank_buffer[idxs], dataset.target_policy_buffer[idxs]

def tee(text, f=None):
    print(text)
    if f is not None:
        f.write(text)
        f.write("\n")
        f.flush()

def plot_weight(model):
    qmodel = model.quantize()
    plt.hist(qmodel.embed.weight.detach().to("cpu").numpy().ravel(), bins=256, range=(-1024.5, 1023.5))
    plt.show()
    plt.hist(qmodel.linear_condition.weight.detach().to("cpu").numpy().ravel(), bins=256, range=(-128.5, 127.5))
    plt.show()
    plt.hist(qmodel.linear_2.weight.detach().to("cpu").numpy().ravel(), bins=256, range=(-128.5, 127.5))
    plt.show()
    plt.hist(qmodel.linear_3.weight.detach().to("cpu").numpy().ravel(), bins=256, range=(-128.5, 127.5))
    plt.show()
    plt.hist(qmodel.linear_4.weight.detach().to("cpu").numpy().ravel(), bins=256, range=(-128.5, 127.5))
    plt.show()
    for name, params in qmodel.named_parameters():
        print(name, params.data.to("cpu").min().numpy(), params.data.to("cpu").max().numpy())


if __name__ == "__main__":
    # 学習

    # 設定
    #torch.set_num_threads(2)
    batch_size = 4096
    #device = "cpu"   # 2.5 it/sec (batch_size = 4096)
    #device = "cuda"  # 30 it/sec (batch_size = 4096)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 250
    sgd_learning_rate = 1e-1
    adam_learning_rate = 1e-3
    N_FEATURES = 2458
    out_dir = Path("./out")
    #kif_files = ["kif.kif1"] * 4096  # TODO

    # 出力ディレクトリ作成
    checkpoint_dir = out_dir / "checkpoint"
    if not out_dir.exists():
        out_dir.mkdir()
        checkpoint_dir.mkdir()

    # データ
    print("loading data...")
    sleep(0.5)
    if "dataset" not in vars():
        #dataset = Dataset(kif_files)
        dataset = EfficientDataset(kif_files)
    print(f"len(dataset)={len(dataset)}")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn, pin_memory=True, drop_last=True
    )
    print("loaded!")
    if "additional_kif_files" in vars():
        print("loading additional data...")
        dataset.load_all_states(additional_kif_files)
        print("loaded!")

    # モデル、最適化手法、損失関数
    model = Model(features=N_FEATURES)
    model.to(device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=sgd_learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=adam_learning_rate, amsgrad=True)
    criterion = Loss()

    # 記録
    epoch_losses = []
    f_log = open(out_dir / "log.txt", "a")

    start_epoch = 0
    
    # チェックポイントの読み込み
    if "old_checkpoint_file" in vars() and Path(old_checkpoint_file).is_file():
        dict_checkpoint = torch.load(old_checkpoint_file, map_location=device)
        print("checkpoint file found!")
        start_epoch = dict_checkpoint["epoch"] + 1
        dict_checkpoint["state_dict"]["linear_4.bias"][0] = 0.0
        model.load_state_dict(dict_checkpoint["state_dict"])
        #optimizer.load_state_dict(dict_checkpoint["optimizer"])

    model.clip_params(soft=True)
    
    # 学習ループ
    for epoch in range(start_epoch, n_epochs):
        if epoch % 10 == 0:
            plot_weight(model)
        
        
        model.train()
        tee(f"epoch {epoch}", f_log)
        sleep(0.5)
        
        epoch_loss = 0.0
        n_predicted_data = 0
        iteration = 0
        for agent_features, condition_features, target_rank, target_policy in tqdm(dataloader):
            agent_features = agent_features.to(device)
            condition_features = condition_features.to(device)
            target_rank = target_rank.to(device)
            target_policy = target_policy.to(device)

            optimizer.zero_grad()
            preds = model(agent_features, condition_features)
            loss = criterion(preds, target_rank, target_policy)
            loss.backward()
            optimizer.step()
            if iteration % 5 == 0:
                model.clip_params()

            epoch_loss += loss.item() * len(agent_features)
            n_predicted_data += len(agent_features)
            iteration += 1

        epoch_loss /= n_predicted_data
        tee(f"epoch_loss = {epoch_loss}", f_log)
        epoch_losses.append(epoch_loss)

        if out_dir is not None and (epoch % 10 == 0 or epoch == n_epochs - 1):
            torch.save({
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "epoch_losses": epoch_losses,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, checkpoint_dir / f"{epoch:03d}.pt")

    f_log.close()


# TODO: 環境チェック