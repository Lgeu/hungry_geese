#pragma once
#include "Stage.hpp"
#include "GreedyAgent.hpp"

namespace hungry_geese {

	struct Agent {
		// コンストラクタ
		Agent();

		// 行動を決定する
		void setActions(Stage& aStage, int aIndex);

		// エージェント
		GreedyAgent Agent0;
		GreedyAgent Agent1;
		GreedyAgent Agent2;
		GreedyAgent Agent3;

		// 行動一覧
		const Actions Idx_to_Actions = {
			Action::NORTH,
			Action::EAST,
			Action::SOUTH,
			Action::WEST,
		};
	};

}