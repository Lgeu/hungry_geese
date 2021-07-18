#pragma once
#include <ctime>

namespace hungry_geese {
    
struct Timer {
    // コンストラクタ
    Timer();

    // 操作
    // タイマー始動
    void start();
    // タイマー停止
    void stop();

    // 問い合わせ
    // 経過秒数を取得
    float elapsedSec() const;

    // メンバ変数
    // 開始時間
    std::clock_t mTimeBegin;
    // 停止時間
    std::clock_t mTimeEnd;
};

}