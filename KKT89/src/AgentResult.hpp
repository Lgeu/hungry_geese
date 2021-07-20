#pragma once
#include "Action.hpp"
#include "library.hpp"

namespace hungry_geese {

struct AgentResult {
    // コンストラクタ
    AgentResult();
    
    // 取る行動
    int mAction; 
    // 盤面評価値
    float mValue;
    // 手の評価
    std::array<float, 4> mPolicy;
    // 特徴量ベクトル
    std::array<nagiss_library::Stack<int, 100>, 4> mAgentFeatures;
    nagiss_library::Stack<int, 100> mConditionFeatures;
};

}