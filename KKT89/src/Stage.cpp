#pragma once
#include "Stage.hpp"

namespace hungry_geese {

Stage::Stage() : mTurn(), mGeese(), mFoods(), mBoard(), mRemainingTime(), mAgentResult() {
    for (int i = 0; i < 77; ++i) {
        mBoard[i] = 0;
    }
    for (int i = 0; i < 4; ++i) {
        mLastActions[i] = hungry_geese::Action::NONE;
    }
}

const Geese& Stage::geese() const {
    return mGeese;
}

const Foods& Stage::foods() const {
    return mFoods;
}

const Actions& Stage::actions() const {
    return mActions;
}

bool Stage::isEnd() const {
    if (mTurn == 199) {
        return true;
    }
    int Survivor = 0;
    for (Goose goose: mGeese) {
        if (goose.isSurvive()) {
            ++Survivor;
        }
    }
    return (Survivor <= 1);
}

void Stage::InitializeBoard() {
    // Goose
    for (Goose goose: geese()) {
        if (!goose.isSurvive()) continue;
        auto items = goose.items();
        for (int i = 0; i < items.right; ++i) {
            auto pos = items[i];
            mBoard[pos.id] = 1;
        }
    }
    // Food
    for (Food food: foods()) {
        auto pos = food.pos();
        mBoard[pos.id] = 1;
    }

    return;
}

bool Stage::isEmpty(int aId) const {
    return (mBoard[aId] == 0);
}

int Stage::randPos(int aId) const {
    for (int i = 0; i < 77; ++i) {
        if (isEmpty(aId)) {
            return aId;
        }
        ++aId;
        if (aId >= 77) {
            aId = 0;
        }
    }
    return aId;
}

}