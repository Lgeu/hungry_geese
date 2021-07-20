#pragma once

namespace hungry_geese {

struct Cpoint {
    // コンストラクタ
    Cpoint();
    Cpoint(int aX, int aY);
    Cpoint(int aId);
    // メンバ変数
    signed char mC;
    // 呼び出し
    int X() const;
    int Y() const;
    int Id() const;
    // 演算子オーバーロード
    Cpoint& operator= (const Cpoint &aPos);
    bool operator== (const Cpoint &aPos) const;
};

}