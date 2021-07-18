#pragma once
#include "MctsAgent.hpp"

namespace hungry_geese {

//------------------------------------------------------------------------------
// コンストラクタ
MctsAgent::MctsAgent() : model(), duct() {}

void MctsAgent::SetTimeLimit(float atimelimit) {
    timelimit = atimelimit;
}

//------------------------------------------------------------------------------
// 実行
AgentResult MctsAgent::run(const Stage& aStage, int aIndex, float timelimit) {
    // 0ターン目は評価値最大の行動をする
    if (aStage.mTurn == 0) {
        return solve1(aStage, aIndex);
    }
    // DUCTをここで実装
    {
        duct.Setprintlog(false);
        AgentResult result;
        duct.InitDuct(aStage, aIndex);
        // 秒数はここで指定
        duct.Search(timelimit);
        auto rootnode = duct.RootNode();
        for (int i = 0; i < 4; ++i) {
            result.mPolicy[i] = (float)rootnode.n[0][i] / (float)(rootnode.n[0][0] + rootnode.n[0][1] + rootnode.n[0][2] + rootnode.n[0][3]);
        }
        result.mValue = rootnode.value[0];
        unsigned char opt_action = 0;
        for (int i = 0; i < 4; ++i) {
            if (result.mPolicy[opt_action] < result.mPolicy[i]) {
                opt_action = i;
            }
        }
        result.mAction = opt_action;
        return result;
    }
}

//------------------------------------------------------------------------------
// 評価値最大の行動を返す
AgentResult MctsAgent::solve1(const Stage& aStage, int aIndex) {
    AgentResult result; 
    std::array<Stack<Point, 77>, 4> geese;
    std::array<Point, 2> foods;
    for (int i = 0; i < 4; ++i) {
        if (!aStage.geese()[i].isSurvive()) {
            continue;
        }
        geese[i] = aStage.geese()[i].items();
    }
    for (int i = 0; i < 2; ++i) {
        foods[i] = aStage.foods()[i].pos();
    }
    std::swap(geese[0], geese[aIndex]);
    auto res = model.evaluate(geese, foods);
    result.mValue = res.value;
    for (int i = 0; i < 4; ++i) {
        result.mPolicy[i] = res.policy[i];
    }
    { // 前ターンの反対の行動を取らない
        auto act = aStage.mLastActions;
        if (act[aIndex] == Action::NORTH) {
            result.mPolicy[2] = -100;
        }
        else if (act[aIndex] == Action::EAST) {
            result.mPolicy[3] = -100;
        }
        else if (act[aIndex] == Action::SOUTH) {
            result.mPolicy[0] = -100;
        }
        else if (act[aIndex] == Action::WEST) {
            result.mPolicy[1] = -100;
        }
    }
    // 評価値の一番高い手を選ぶ
    int opt_action = 0;
    for (int i = 0; i < 4; ++i) {
        if (result.mPolicy[opt_action] < result.mPolicy[i]) {
            opt_action = i;
        }
    }
    result.mAction = opt_action;
    return result;
}

//------------------------------------------------------------------------------
// 方向を指定して移動先を返す関数
Point MctsAgent::Translate(Point aPos, int Direction) {
    int nx = aPos.x;
    int ny = aPos.y;
    nx += dx[Direction];
    if (nx < 0) {
        nx += Parameter::rows; 
    }
    if (nx == Parameter::rows) {
        nx = 0;
    }
    ny += dy[Direction];
    if (ny < 0) {
        ny += Parameter::columns; 
    }
    if (ny == Parameter::columns) {
        ny = 0;
    }
    return Point(nx,ny);
}

}