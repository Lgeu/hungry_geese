from pprint import pprint
from typing import List
from pathlib import Path
from dataclasses import dataclass, asdict


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
        agent_features: List[List[int]]
        condition_features: List[int]

    format_version: str
    kif_id: str
    seed: int
    agent_information: List[str]
    steps: List[Step]
    ranks: List[int]

    def print(self):
        print("format_version:", self.format_version)
        print("kif_id")

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
            format_version = f.readline().strip()
            kif_id = f.readline().strip()
            seed = int(f.readline().strip())
            agent_information = [f.readline().strip() for _ in range(4)]
            steps = []
            while True:
                step = int(f.readline().strip())
                remaining_times = list(map(float, f.readline().strip().split()))
                agent_positions = []
                for _ in range(4):
                    _, *positions = map(int, f.readline().strip().split())
                    agent_positions.append(positions)
                food_positions = list(map(int, f.readline().strip().split()))
                moves = list(map(int, f.readline().strip().split()))
                values = [list(map(float, f.readline().strip().split())) for _ in range(4)]
                agent_features = [list(map(int, f.readline().strip().split()))[1:] for _ in range(4)]
                _, *condition_features = map(int, f.readline().strip().split())
                steps.append({
                    "step": step,
                    "remaining_times": remaining_times,
                    "agent_positions": agent_positions,
                    "food_positions": food_positions,
                    "moves": moves,
                    "values": values,
                    "agent_features": agent_features,
                    "condition_features": condition_features,
                })
                if all(m == -100 for m in moves):
                    break
            ranks = list(map(int, f.readline().strip().split()))
            return cls.from_dict({
                "format_version": format_version,
                "kif_id": kif_id,
                "seed": seed,
                "agent_information": agent_information,
                "steps": steps,
                "ranks": ranks,
            })


def test_kif_read():
    kif = Kif.from_file("./KKT89/src/out/20210712153013_358084156.kif1")
    pprint(asdict(kif))


if __name__ == "__main__":
    test_kif_read()
