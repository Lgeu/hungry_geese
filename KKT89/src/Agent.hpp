#pragma once
#include "Stage.hpp"
#include "GreedyAgent.hpp"
#include "MctsAgent.hpp"

namespace hungry_geese {

struct Agent {
    // コンストラクタ
    Agent();

    // 行動を決定する
    void setActions(Stage& aStage, int aIndex, float timelimit);

    // エージェント
    MctsAgent Agent0;
    MctsAgent Agent1;
    MctsAgent Agent2;
    MctsAgent Agent3;

    // 行動一覧
    const Actions Idx_to_Actions = {
        Action::NORTH,
        Action::EAST,
        Action::SOUTH,
        Action::WEST,
    };
};

}