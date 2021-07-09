#pragma once
#include "Action.hpp"
#include "Stack.hpp"

namespace hungry_geese {

struct AgentResult {
	// コンストラクタ
	AgentResult();
	
	// 取る行動
	Action mAction; 
	// 盤面評価値
	float mValue;
	// 手の評価
	std::array<float, 4> mPolicy;
	// 特徴量ベクトル
	Stack<int, 100> mFeatures;
};

}