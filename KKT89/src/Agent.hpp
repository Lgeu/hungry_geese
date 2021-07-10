#pragma once
#include "Stage.hpp"
#include "GreedyAgent.hpp"

namespace hungry_geese {

namespace Agent0 {
	GreedyAgent Agent;
}

namespace Agent1 {
	GreedyAgent Agent;
}

namespace Agent2 {
	GreedyAgent Agent;
}

namespace Agent3 {
	GreedyAgent Agent;
}

struct Agent {
	// コンストラクタ
	Agent();

	// 行動を決定する
	void setActions(Stage& aStage, int aIndex);

	// 行動一覧
	const Actions Idx_to_Actions = {
		Action::NORTH,
		Action::EAST,
		Action::SOUTH,
		Action::WEST,
	};
};

}