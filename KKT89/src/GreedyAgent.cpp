#pragma once
#include "GreedyAgent.hpp"

namespace hungry_geese {

GreedyAgent::GreedyAgent() {}

AgentResult GreedyAgent::run(const Stage& aStage, int aIndex) {
    AgentResult result;
    std::array<int, 4> can_actions;
    for (int i = 0; i < 4; ++i) {
        can_actions[i] = 1;
    }
    { // 前ターンの反対の行動を取らない
        auto act = aStage.mLastActions;
        if (act[aIndex] == Action::NORTH) {
            can_actions[2] = 0;
        }
        else if (act[aIndex] == Action::EAST) {
            can_actions[3] = 0;
        }
        else if (act[aIndex] == Action::SOUTH) {
            can_actions[0] = 0;
        }
        else if (act[aIndex] == Action::WEST) {
            can_actions[1] = 0;
        }
    }
    // 現在身体のある位置に移動しない
    for (int i = 0; i < 4; ++i) {
        auto Pos = Translate(aStage.geese()[aIndex].items()[0], i);
        for (int j = 0; j < 4; ++j) {
            if (!aStage.geese()[j].isSurvive()) {
                continue;
            }
            for (auto aPos:aStage.geese()[j].items()) {
                if (Pos == aPos) {
                    can_actions[i] = 0;
                }
            }
        }
    }
    // 相手の頭の隣接4マスには移動しない
    for (int i = 0; i < 4; ++i) {
        auto Pos = Translate(aStage.geese()[aIndex].items()[0], i);
        for (int j = 0; j < 4; ++j) {
            if (aIndex == j) {
                continue;
            }
            if (!aStage.geese()[j].isSurvive()) {
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                auto aPos = Translate(aStage.geese()[j].items()[0], k);
                if (Pos == aPos) {
                    can_actions[i] = 0;
                }
            }
        }
    }
    int opt_action = 0;
    for (int i = 0; i < 4; ++i) {
        if (can_actions[i]) {
            opt_action = i;
            break;
        }
    }
    // 食べ物に一番近い位置に移動する
    int min_food_distance = INF;
    for (int i = 0; i < 4; ++i) {
        if (can_actions[i]) {
            for (int j = 0; j < 2; ++j) {
                int result = min_Distance(aStage.foods()[j].pos(), Translate(aStage.geese()[aIndex].items()[0], i));
                if (min_food_distance > result) {
                    min_food_distance = result;
                    opt_action = i;
                }
            }
        }
    }
    result.mAction = opt_action;
    return result;
}

int GreedyAgent::min_Distance(Point aPos, Point bPos) {
    int result = 0;
    int row = std::abs(aPos.x - bPos.x);
    result += std::min(row, Parameter::rows - row);
    int column = std::abs(aPos.y - bPos.y);
    result += std::min(column, Parameter::columns - column);
    return result;
}

Point GreedyAgent::Translate(Point aPos, int Direction) {
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