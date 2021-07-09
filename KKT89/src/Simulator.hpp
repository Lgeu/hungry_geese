#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include "Random.hpp"
#include "Game.hpp"
#include "Parameter.hpp"
#include "Agent.hpp"

namespace hungry_geese {
    
struct Simulator {
    // コンストラクタ
    Simulator();

    // ゲームのSEED値を変更
    void changeSeed(uint aX);
    // 日付文字列を取得
    std::string getDatetimeStr() const;
    // 棋譜IDを設定
    void setKifID();

    // ゲームの実行
    void run();

    // 棋譜の出力
    void printKif() const;

    // ゲーム全体
    Game mGame;
    // Agent
    Agent mAgent;

    // 乱数
    Random rand;
    // 棋譜ID(6桁)
    std::string KifID = "000000";
    // 棋譜拡張子
    const std::string KifExtension = ".kif1";
};

}