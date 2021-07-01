#pragma once
#include "Stage.hpp"

namespace hungry_geese {

// ステージの配列
using Stages = std::array<Stage, 200>;

struct Game {
	// コンストラクタ
	Game();

	// 問い合わせ
	// 現在のターン数
	int turn() const;

	// メンバ変数
	// 現在のターン数
	int mTurn;
	// ステージの配列
	Stages mStages;
};

}

// かなり未実装