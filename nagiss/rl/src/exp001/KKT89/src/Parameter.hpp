#pragma once

namespace hungry_geese {
    
struct Parameter {
    // ターン数
    static const int episodeSteps = 200;
    // 考慮時間
    static const int actTimeout = 1;
    // 持ち時間
    static const int remainingOverageTime = 60;
    // ステージの横幅
    static const int columns = 11;
    // ステージの縦幅
    static const int rows = 7;
    // 各ターン存在する食べ物の数
    static const int min_food = 2;
    // max_length
    static const int max_length = 99;
    // hunger_rate 毎に長さが減少
    static const int hunger_rate = 40;
};

}