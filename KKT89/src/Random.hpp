#pragma once
#include <random>
#include "Types.hpp"

namespace hungry_geese {

struct Random {
    std::mt19937 engine;

    // 特定の乱数シードで初期化するコンストラクタ
    Random(uint aX = 0);

    // 乱数生成
    // [0, aTerm) の範囲で乱数を生成する
    int randTerm(int aTerm);
    // [aMin, aTerm) の範囲で乱数を生成する
    int randMinTerm(int aMin, int aTerm);
    // [aMin, aMax] の範囲で乱数を生成
    int randMinMax(int aMin, int aMax);

    // 乱数シード
    uint x = 0;
};

}