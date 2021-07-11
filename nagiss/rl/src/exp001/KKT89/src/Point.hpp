#pragma once

namespace hungry_geese {
    
struct Point {
    // コンストラクタ
    Point();
    Point(int aX, int aY);
    Point(int aId);

    // メンバ変数
    int x;
    int y;
    int id;

    // 演算子オーバーロード
    bool operator== (const Point &aPos) const;
};

}