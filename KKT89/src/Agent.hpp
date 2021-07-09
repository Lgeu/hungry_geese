#pragma once
#include "Stage.hpp"
#include <cmath>

namespace hungry_geese {

// 行動の配列
using Actions = std::array<Action, 4>;

struct Agent {
	// コンストラクタ
	Agent();

	// 行動を決定する
	void setActions(Stage& aStage);
	void setActions(Stage& aStage, int aIndex);
	void GreedyAgent(Stage& aStage, int aIndex);

	// 関数
	int min_Distance(Point aPos, Point bPos);
	Point Translate(Point aPos, int Direction);

	// 行動一覧
	const Actions Idx_to_Actions = {
		Action::NORTH,
		Action::EAST,
		Action::SOUTH,
		Action::WEST,
	};
	// いつもの
	const std::array<int, 4> dx = {-1, 0, 1, 0};
	const std::array<int, 4> dy = {0, 1, 0, -1};
	const int INF = 1e9;
};

}