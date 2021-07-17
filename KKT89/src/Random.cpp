#pragma once
#include "Random.hpp"
#include "Assert.hpp"

namespace hungry_geese {

Random::Random(uint aX) {
    // 指定がない時ランダムに変更…
    if (aX == -1) {
        Random::x = std::chrono::steady_clock::now().time_since_epoch().count();
    }
    else {
        Random::x = aX;
    }
    std::mt19937 aengine(Random::x);
    Random::engine = aengine;
}

int Random::randTerm(int aTerm) {
    return Random::engine() % aTerm;
}

int Random::randMinTerm(int aMin, int aTerm) {
    return aMin + randTerm(aTerm - aMin);
}

int Random::randMinMax(int aMin, int aMax) {
    return aMin + randTerm(1 + aMax - aMin);
}

}