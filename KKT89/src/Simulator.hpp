#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include "Random.hpp"
#include "Game.hpp"
#include "Parameter.hpp"

namespace hungry_geese {
    
struct Simulator {
    // コンストラクタ
    Simulator();

    // ゲームのSEED値を変更
    void changeSeed(uint aX);
    // 棋譜IDを設定(6桁)
    void setKifID(int idx);

    // ゲームの実行
    void run();

    // 棋譜の出力
    void printKif() const;

    // ゲーム全体
    Game mGame;

    // 乱数
    Random rand;
    // 棋譜ID(6桁)
    std::string KifID = "000000";
    // 棋譜拡張子
    const std::string KifExtension = ".kif1";
};

}