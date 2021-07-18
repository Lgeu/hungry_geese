#pragma once
#include "Stage.hpp"
#include <algorithm>

namespace hungry_geese {

// ステージの配列
using Stages = std::array<Stage, 200>;
// reward
using Rewards = std::array<int, 4>;

struct Game {
    // コンストラクタ
    Game();

    // 問い合わせ
    // 現在のターン数
    int turn() const;

    // 順位確定
    void calc_Ranking();

    // メンバ変数
    // 現在のターン数
    int mTurn;
    // ステージの配列
    Stages mStages;
    // rewardの配列
    Rewards mRewards;
    // 順位
    std::array<int, 4> mStanding;
};

}