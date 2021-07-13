import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

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
            torch.Tensor: condition_features 状態特徴       (length,)   dtype=long
            torch.Tensor: target_rank        最終順位       (4,)        dtype=long
            torch.Tensor: target_policy      探索で得た方策  (4, 4)      dtype=float
        """

        kif = self.kifs[idx]
        n_steps = len(kif.steps) - 1  # 最後のステップは着手が無いので除外
        step = torch.randint(n_steps, tuple())
        agent_features = pad_sequence([torch.tensor(feats) for feats in kif.steps[step].agent_features], batch_first=True, padding_value=-100)
        condition_features = torch.tensor(kif.steps[step].condition_features)
        target_rank = torch.tensor(kif.ranks)
        target_policy = torch.tensor(kif.steps[step].values, dtype=torch.float)[:, 1:]
        assert target_policy.shape == (4, 4)
        return agent_features, condition_features, target_rank, target_policy

    def __len__(self):
        return len(self.kif_files)


class Model(nn.Module):
    def __init__(self, features, out_dim=5, hidden_1=256, hidden_2=32):
        super().__init__()
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.embed = nn.EmbeddingBag(features, hidden_1, mode="sum", padding_idx=-100)
        self.embed.weight.data /= 8.0  # 小さめの値で初期化
        self.linear_condition = nn.Linear(hidden_1, hidden_2)
        self.linear_2 = nn.Linear(hidden_1, hidden_2)
        self.linear_3 = nn.Linear(hidden_2, hidden_2)
        self.linear_4 = nn.Linear(hidden_2, out_dim)

    def forward(self, x, condition):
        """

        Args:
            x: [[1, 3, 9, ..., 22, -100, -100], ... ] みたいなテンソル (batch, 4, length)
            condition: (batch, length)
        Returns:
            torch.Tensor: (batch, 4, out_dim)
        """
        batch_size = x.shape[0]

        # (1) [batch, 4, length] -> [batch, 4, 256]
        x = F.relu_(self.embed(x.view(batch_size * 4, -1)).view(batch_size, 4, self.hidden_1))

        # (2) [batch, length] -> [batch, 256]
        condition = F.relu_(self.embed(condition))
        condition += x.sum(1)

        # (3) [batch, 256] -> [batch, 32]
        condition = F.relu_(self.linear_condition(condition))

        # (4) [batch, 4, 256] -> [batch, 4, 32]
        x = F.relu_(self.linear_2(x))
        x += condition.unsqueeze(1)

        # (5) [batch, 4, 32] -> [batch, 4, 32]
        x = F.relu_(self.linear_3(x))

        # (6) [batch, 4, 32] -> [batch, 4, out_dim]
        x = self.linear_4(x)

        return x


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
        value_loss = 0.0
        for a in range(4):
            for b in range(a+1, 4):
                rank_diff = target_rank[:, a] - target_rank[:, b]  # (batch,)
                t = torch.where(rank_diff < 0, torch.tensor(1.0), torch.tensor(0.0))
                t = torch.where(rank_diff == 0, torch.tensor(0.5), t)
                pred = x[:, a, 0] - x[:, b, 0]
                value_loss = value_loss + F.binary_cross_entropy_with_logits(pred, t)
        value_loss /= 6.0  # 4C2

        pred = x[:, :, 1:].reshape(batch_size * 4, 4)
        alive = target_policy[:, :, 0] != -100.0  # (batch, 4)
        target = (target_policy*alive.unsqueeze(2)).view(batch_size*4, 4)
        policy_loss = soft_cross_entropy(pred, target) * 0.5 / (alive.sum()+1e-10)

        #print("loss:", value_loss.item(), policy_loss.item())
        loss = value_loss+policy_loss
        return loss
