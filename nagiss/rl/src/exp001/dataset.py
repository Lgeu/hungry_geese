from typing import List
from pathlib import Path
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Kif:
    @dataclass
    class Step:
        step: int
        remaining_times: List[float]
        agent_positions: List[List[int]]
        food_positions: List[int]
        moves: List[int]
        values: List[List[float]]

    format_version: int
    kif_id: str
    seed: int
    agent_information: List[str]
    steps: List[Step]
    ranks: List[int]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["steps"] = [cls.Step(**step) for step in d["steps"]]
        return Kif(**d)

    @classmethod
    def from_file(cls, filename):
        """
        Args:
            filename (str|Path): 棋譜ファイル

        Returns:

        """
        with open(filename) as f:
            format_version = f.readline()
            kif_id = f.readline()
            seed = int(f.readline())
            agent_information = [f.readline() for _ in range(4)]
            steps = []
            while True:
                step = int(f.readline())
                remaining_times = list(map(float, f.readline().split()))
                agent_positions = []
                for _ in range(4):
                    _, positions = map(int, f.readline().split())
                    agent_positions.append(positions)
                food_positions = list(map(int, f.readline().split()))
                moves = list(map(int, f.readline().split()))
                values = [list(map(float, f.readline().split())) for _ in range(4)]
                steps.append({
                    "step": step,
                    "remaining_times": remaining_times,
                    "agent_positions": agent_positions,
                    "food_positions": food_positions,
                    "moves": moves,
                    "values": values,
                })
                if all(m == -100 for m in moves):
                    break
            ranks = list(map(int, f.readline().split()))
            return cls.from_dict({
                "format_version": format_version,
                "kif_id": kif_id,
                "seed": seed,
                "agent_information": agent_information,
                "steps": steps,
                "ranks": ranks,
            })


class BitBoard:
    mask_right = 0b10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000
    mask_left = 0b00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001
    mask_rights = [
        0,
        0b10000000000_10000000000_10000000000_10000000000_10000000000_10000000000_10000000000,
        0b11000000000_11000000000_11000000000_11000000000_11000000000_11000000000_11000000000,
        0b11100000000_11100000000_11100000000_11100000000_11100000000_11100000000_11100000000,
        0b11110000000_11110000000_11110000000_11110000000_11110000000_11110000000_11110000000,
        0b11111000000_11111000000_11111000000_11111000000_11111000000_11111000000_11111000000,
        0b11111100000_11111100000_11111100000_11111100000_11111100000_11111100000_11111100000,
    ]
    mask_lefts = [
        0,
        0b00000000001_00000000001_00000000001_00000000001_00000000001_00000000001_00000000001,
        0b00000000011_00000000011_00000000011_00000000011_00000000011_00000000011_00000000011,
        0b00000000111_00000000111_00000000111_00000000111_00000000111_00000000111_00000000111,
        0b00000001111_00000001111_00000001111_00000001111_00000001111_00000001111_00000001111,
        0b00000011111_00000011111_00000011111_00000011111_00000011111_00000011111_00000011111,
        0b00000111111_00000111111_00000111111_00000111111_00000111111_00000111111_00000111111,
    ]
    mask_up = (1<<11) - 1
    mask_down = (1<<77) - (1<<66)
    whole = (1<<77) - 1

    def __init__(self, data=0):
        self.data = data

    def __str__(self):
        return "\n".join(format(self.data>>i&(1<<11)-1, "011b")[::-1] for i in range(0, 77, 11))

    def print(self):
        assert 0 <= self.data <= BitBoard.whole
        print(self)

    def set_0(self, idx):
        self.data ^= self.data & (1<<idx)

    def set_1(self, idx):
        self.data |= 1<<idx

    def right(self):  # すべてのビットをひとつ右に動かしたものを返す
        masked = self.data & BitBoard.mask_right
        return BitBoard((self.data ^ masked) << 1 | masked >> 10)

    def left(self):  # すべてのビットをひとつ左に動かしたものを返す
        masked = self.data & BitBoard.mask_left
        return BitBoard((self.data ^ masked) >> 1 | masked << 10)

    def up(self):  # すべてのビットをひとつ上に動かしたものを返す
        masked = self.data & BitBoard.mask_up
        return BitBoard(self.data >> 11 | masked << 66)

    def down(self):  # すべてのビットをひとつ下に動かしたものを返す
        masked = self.data & BitBoard.mask_down
        return BitBoard((self.data ^ masked) << 11 | masked >> 66)

    def slide(self, y, x):
        if y > 0:  # 下
            mask = (1<<77) - (1<<y*11)
            masked = self.data & mask
            self.data ^= masked
            self.data <<= 11 * y
            self.data |= masked >> 77 - 11 * y
        if y < 0:  # 上
            mask = (1<<-11*-y) - 1
            masked = self.data & mask
            self.data >>= -11 * y
            self.data |= masked << 77 + 11 * y
        if x > 0:  # 右
            mask = BitBoard.mask_rights[x]
            masked = self.data & mask
            self.data ^= masked
            self.data <<= x
            self.data |= masked >> 11 - x
        if x < 0:  # 左
            mask = BitBoard.mask_lefts[-x]
            masked = self.data & mask
            self.data ^= masked
            self.data >>= -x
            self.data |= masked << 11 + x

    def invert(self):  # 反転
        self.data ^= BitBoard.whole

    def popcount(self):
        return bin(self.data).count("1")

    def count_right_zero(self):  # 1 になってる場所をひとつ返す  # すべて 0 の場合の動作は考慮しない
        return (self.data & -self.data).bit_length() - 1

    def pext(self, mask):  # C++ なら pext 命令を使う
        assert mask < 1<<55  # C++ で実装したときの 1 個目のデータが保持している範囲で
        idx = 0
        res = 0
        while mask != 0:
            rz = (mask & -mask).bit_length() - 1
            mask ^= 1 << rz
            res |= ((self.data >> rz) & 1) << idx
            idx += 1
        return res

    def copy(self):
        return BitBoard(self.data)


class State:
    def __init__(self, kif, step):
        """
        Args:
            kif (Kif):
            step (int):
        """
        if step >= len(kif.steps) - 1:
            raise ValueError(f"step がでかすぎる step={step} len(kif.steps)={len(kif.steps)}")

        current_agent_bitboards = []  # 4 エージェント + 全体  # これもしかして全体だけでいいのでは…？
        board_all = BitBoard()
        for positions in kif.steps[step].agent_positions:
            board = BitBoard(sum(1 << position for position in positions))
            current_agent_bitboards.append(board)
            board_all.data |= board.data
        current_agent_bitboards.append(board_all)

        # 1 から 5 ターン後の盤面のビットボード
        future_agent_bitboards = [current_agent_bitboards]
        for future in range(1, 6):
            boards = []
            board_all = BitBoard()
            for idx_agent, positions in enumerate(kif.steps[step].agent_positions):
                if len(positions) < future:
                    boards.append(BitBoard())
                else:
                    board = future_agent_bitboards[-1][idx_agent].copy()
                    board.data ^= 1 << positions[-future]
                    boards.append(board)
                    board_all.data |= board
            boards.append(board_all)
            future_agent_bitboards.append(boards)

        self.current_agent_bitboards = current_agent_bitboards
        self.future_agent_bitboards = future_agent_bitboards
        self.step_info = kif.steps[step]
        self.agent_features = [[] for _ in range(4)]

    def extract_features(self):
        from feature_index import dict_agent_feature_to_index
        for idx_agent in range(4):
            features = self.agent_features[idx_agent]  # ここに書き込む
            positions = self.step_info.agent_positions[idx_agent]
            if len(positions) == 0:
                continue
            head_position = positions[0]
            tail_position = positions[-1]

            # 距離 2 までの全パターン
            current_board = self.current_agent_bitboards[-1]
            assert current_board.data >> head_position & 1




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
    def __init__(self, agent_features, condition_features, out_dim=5, hidden_1=256, hidden_2=32):
        super().__init__()
        # TODO: EmbeddingBag に書き換える
        # TODO: Embedの初期化を小さめの値にする
        self.embed = nn.Embedding(agent_features, hidden_1, padding_idx=-100)
        self.embed_condition = nn.Embedding(condition_features, hidden_1, padding_idx=-100)
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
        # [batch, 4, length] -> [batch, 4, length, 256]
        x = self.embed(x)
        # [batch, 4, length, 256] -> [batch, 4, 256]
        x = F.relu_(x.sum(2))

        # [batch, length] -> [batch, length, 256]
        condition = self.embed_condition(condition)
        # [batch, length, 256] -> [batch, 256]
        condition = F.relu_(condition.sum(1))
        condition += x.sum(1)
        # [batch, 256] -> [batch, 32]
        condition = F.relu_(self.linear_condition(condition))

        # [batch, 4, 256] -> [batch, 4, 32]
        x = self.linear_2(x)
        x += condition.unsqueeze(1)
        x = F.relu_(x)

        # [batch, 4, 32] -> [batch, 4, 32]
        x = F.relu_(self.linear_3(x))

        # [batch, 4, 32] -> [batch, 4, out_dim]
        x = self.linear_4(x)

        return x


class Loss(nn.Module):
    def __init__(self, elmo_lambda=0.5):
        super().__init__()
        self.elmo_lambda = elmo_lambda

    def forward(self, x, target_rank, target_move, target_value, alive):
        """

        Args:
            x: (batch, 4, 5)
            target_rank (torch.LongTensor): (batch, 4)
            target_move (torch.LongTensor): (batch, 4)
            target_value: (batch, 4, 5)  探索で得た値
            alive: (batch, 4)

        Returns:

        """
        batch_size = x.shape[0]
        value_loss = 0.0
        for a in range(4):
            for b in range(a+1, 4):
                rank_diff = target_rank[:, a] - target_rank[:, b]  # (batch,)
                t = torch.where(rank_diff < 0, 1.0, 0.0).to(torch.float32)
                t = torch.where(rank_diff == 0, 0.5, t)
                pred = x[:, a, 0] - x[:, b, 0]
                value_loss = value_loss + F.binary_cross_entropy_with_logits(pred, t) \
                             + self.elmo_lambda * F.binary_cross_entropy_with_logits(pred, target_value[:, a, 0] - target_value[:, b, 0])
        value_loss /= 6.0  # 4C2

        pred = x[:, :, 1:].reshape(batch_size * 4, 4)[alive.view(-1)]
        move = target_move.view(batch_size * 4)[alive.view(-1)]
        target = target_value[:, :, 1:].reshape(batch_size * 4, 4)[alive.view(-1)]
        policy_loss = F.cross_entropy(pred, move) \
                      + self.elmo_lambda * F.binary_cross_entropy_with_logits(pred, target)
        print("loss:", value_loss, policy_loss)
        loss = value_loss + policy_loss
        return loss







