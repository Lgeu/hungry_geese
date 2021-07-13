import torch
from torch import nn
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, kif_files):
        """
        Args:
            kif_files (list[str]): 棋譜ファイル名のリスト
        """

        self.kif_files = kif_files

    def __getitem__(self, item):
        """
        NN への入力と、
        4 プレイヤー分の [
            盤面評価値, 手1の評価値, 手2の評価値, 手3の評価値, 手4の評価値
        ] と、最終順位と実際の着手を返す

        Args:
            item (int):
        Returns:
            torch.Tensor:
            torch.Tensor:
        """

        # TODO

    def __len__(self):
        return len(self.kif_files)


# 全員の評価値を同時に出すには…？
# 自分と自分以外の区別だけしたモデルに
# 元のモデルは頭と尻尾の関係をうまく捉えられないのでは？

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
    """クロスエントロピー

    全バッチの合計

    Args:
        pred: softmax 前の値 (batch, n)
        target: (batch, n)

    Returns:

    """
    return -(target*F.log_softmax(pred, dim=1)).sum()


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target_rank, target_policy, alive):
        """

        Args:
            x: (batch, 4, 5)
            target_rank (torch.LongTensor): (batch, 4)
            target_policy (torch.tensor): (batch, 4, 4)  探索で得た値
            alive: (batch, 4)  01 変数

        Returns:

        """
        batch_size = x.shape[0]
        value_loss = 0.0
        for a in range(4):
            for b in range(a+1, 4):
                rank_diff = target_rank[:, a]-target_rank[:, b]  # (batch,)
                t = torch.where(rank_diff < 0, torch.tensor(1.0), torch.tensor(0.0))
                t = torch.where(rank_diff == 0, torch.tensor(0.5), t)
                pred = x[:, a, 0]-x[:, b, 0]
                value_loss = value_loss+F.binary_cross_entropy_with_logits(pred, t)
        value_loss /= 6.0  # 4C2

        pred = x[:, :, 1:].reshape(batch_size*4, 4)
        target = target_policy.view(batch_size*4, 4)*alive.view(-1)
        policy_loss = soft_cross_entropy(pred, target) * 0.5 / (alive.sum()+1e-10)

        print("loss:", value_loss.item(), policy_loss.item())
        loss = value_loss+policy_loss
        return loss
