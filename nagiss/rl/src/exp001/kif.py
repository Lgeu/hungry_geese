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

    def kaggle_env_steps(self):
        class Struct(dict):
            def __init__(self, **entries):
                entries = {k: v for k, v in entries.items() if k != "items"}
                dict.__init__(self, entries)
                self.__dict__.update(entries)
            def __setattr__(self, attr, value):
                self.__dict__[attr] = value
                self[attr] = value
        res = []
        for step in self.steps:
            agents = [Struct() for _ in range(4)]
            for i, agent in enumerate(agents):
                agent.action = "NORTH"
                agent.reward = 0
                agent.info = {}
                agent.observation = Struct(remainingOverageTime=60, index=i)
                agent.status = "ACTIVE" if len(step.agent_positions[i]) else "DONE"
            agents[0].observation.step = step.step
            agents[0].observation.geese = step.agent_positions
            agents[0].observation.food = step.food_positions
            res.append(agents)
        return res


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


from bisect import bisect_right

FEATURE_NAMES = """
    NEIGHBOR_UP_7,
    NEIGHBOR_DOWN_7,
    NEIGHBOR_LEFT_7,
    NEIGHBOR_RIGHT_7,
    LENGTH,
    DIFFERENCE_LENGTH_1ST,
    DIFFERENCE_LENGTH_2ND,
    DIFFERENCE_LENGTH_3RD,
    DIFFERENCE_LENGTH_4TH,
    RELATIVE_POSITION_TAIL,
    RELATIVE_POSITION_OPPONENT_HEAD,
    RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL,
    RELATIVE_POSITION_FOOD,
    MOVE_HISTORY,
    RELATIVE_POSITION_TAIL_ON_PLANE_X,
    RELATIVE_POSITION_TAIL_ON_PLANE_Y,
    N_REACHABLE_POSITIONS_WITHIN_1_STEP,
    N_REACHABLE_POSITIONS_WITHIN_2_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_3_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_4_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_5_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_6_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_7_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_8_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_2_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_3_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_4_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_5_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_6_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_7_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_8_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_2_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_3_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_4_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_5_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_6_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_7_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_8_STEPS,
    N_ALIVING_GEESE,
    N_OCCUPIED_POSITIONS,
    STEP,
    END
""".replace(" ", "").replace("\n", "").split(",")


def interpret_feature(feature):
    BOUNDARY = [0, 128, 256, 384, 512, 589, 610, 631, 652, 673, 750, 826, 902, 978, 1234, 1295, 1356, 1362, 1376, 1402,
                1442, 1496, 1562, 1636, 1714, 1718, 1722, 1726, 1730, 1734, 1738, 1742, 1746, 1752, 1766, 1792, 1832,
                1886, 1952, 2026, 2104, 2107, 2183, 2382]
    OFFSET = [0, 128, 256, 384, 511, 599, 620, 641, 662, 673, 749, 825, 901, 978, 1264, 1325, 1356, 1362, 1376, 1402,
              1442, 1496, 1562, 1636, 1714, 1718, 1722, 1726, 1730, 1734, 1738, 1742, 1746, 1752, 1766, 1792, 1832,
              1886, 1952, 2026, 2102, 2105, 2183, ]
    feature_type = bisect_right(BOUNDARY, feature) - 1
    assert 0 <= feature_type < len(OFFSET), f"feature={feature}, feature_type={feature_type}"
    feature_value = feature - OFFSET[feature_type]
    res = f"{FEATURE_NAMES[feature_type]} = {feature_value} ({bin(feature_value)})"
    return res


def test_kif_read(file="./KKT89/src/out/20210712175538_1926312680.kif1"):
    kif = Kif.from_file(file)
    pprint(asdict(kif))
    for step in kif.steps:
        print(f"# step {step.step}")
        for idx_agent, features in enumerate(step.agent_features):
            print(f"- agent {idx_agent}")
            for f in features:
                print("  -", interpret_feature(f))
        print("- condition")
        for f in step.condition_features:
            print("  -", interpret_feature(f))


if __name__ == "__main__":
    test_kif_read()
