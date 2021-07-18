#pragma once
#include "AgentResult.hpp"
#include "Stage.hpp"
#include "Stack.hpp"
#include "Assert.hpp"
#include "Evaluation_function.hpp"
#include "Duct.hpp"

namespace hungry_geese {

struct MctsAgent {
    // コンストラクタ
    MctsAgent();

    Evaluator model;
    Duct duct; 

    // 制限時間
    float timelimit;
    void SetTimeLimit(float atimelimit);

    // 実行
    AgentResult run(const Stage& aStage, int aIndex, float timelimit);

    // 評価値最大の行動を返す
    AgentResult solve1(const Stage& aStage, int aIndex);

    // 方向を指定して移動先を返す関数
    static Point Translate(Point aPos, int Direction);

    // いつもの
    static constexpr std::array<int, 4> dx = {-1, 0, 1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};

    // 行動一覧
    static constexpr Actions Idx_to_Actions = {
        Action::NORTH,
        Action::EAST,
        Action::SOUTH,
        Action::WEST,
    };
};

}