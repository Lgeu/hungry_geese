#pragma once
#include "Goose.hpp"

namespace hungry_geese {

Goose::Goose() : mItems(), mIsSurvive(true) {
    mRemainingTime = Parameter::remainingOverageTime;
}

Goose::Goose(Point aPos) : mIsSurvive(true) {
    mItems.push(aPos);
    mRemainingTime = Parameter::remainingOverageTime;
}

Items Goose::items() const {
    return mItems;
}

bool Goose::isSurvive() const {
    return mIsSurvive;
}

void Goose::setIsSurvive(bool isSurvive) {
    mIsSurvive = isSurvive;
    if (!isSurvive) {
        mItems.clear();
    }
}

float Goose::remainingTime() const {
    return mRemainingTime;
}

}