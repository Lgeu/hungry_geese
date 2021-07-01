#pragma once
#include "Point.hpp"

namespace hungry_geese {

struct Food {
    // コンストラクタ
    Food();
    Food(Point aPos);

    // 問い合わせ
    // 食べ物の位置
    const Point& pos() const;
    // 既に食べられたか
    bool isEaten() const;
    // 食べられたかどうかを設定する
    void setIsEaten(bool isEaten);

    // メンバ変数
    // 位置
    Point mPosition;
    // 食べられたかどうか
    bool mIsEaten;
};

}