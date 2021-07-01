#pragma once
#include "Simulator.hpp"

namespace hungry_geese {
    
Simulator::Simulator() : mGame() {
    rand = Random(0);
}

void Simulator::changeSeed(uint aX) {
    rand = Random(aX);
}

void Simulator::setKifID(int idx) {
    std::string ID = "000000";
    for (int i = 0; i < 6; ++i) {
        ID[5 - i] = (char)(idx % 10 + '0');
        idx /= 10;
    }
    KifID = ID;
}

void Simulator::run() {
    // 初期条件の設定(未実装)

    // ゲーム終了までターンを進行させる(未実装)

    return;
}

void Simulator::printKif() const {
    std::string filename = "./src/out/" + KifID + KifExtension;
    std::fstream file_out;

    file_out.open(filename, std::ios_base::out);

    if (!file_out.is_open()) {
        // ファイルが開けなかった
        std::cout << "failed to open " << filename << std::endl;
        exit(1);
    }

    // 棋譜ファイルのフォーマットのバージョン
    file_out << KifExtension << std::endl;
    // 棋譜ID
    file_out << KifID << std::endl;
    // 乱数のシード値
    file_out << rand.x << std::endl;
    // エージェント1の情報(string)
    file_out << "test1" << std::endl;
    // エージェント2の情報(string)
    file_out << "test2" << std::endl;
    // エージェント3の情報(string)
    file_out << "test3" << std::endl;
    // エージェント4の情報(string)
    file_out << "test4" << std::endl;

    // 各ステップの出力
    for (int step = 0; step <= mGame.turn(); ++step) {
        auto stage = mGame.mStages[step];

        // ステップ数
        file_out << step << std::endl;

        // エージェントの位置
        for (Goose goose: stage.geese()) {
            auto items = goose.items();
            file_out << items.right;
            for (int i = 0; i < items.right; ++i) {
                auto pos = items[i];
                file_out << " " << pos.id;
            }
            file_out << std::endl;
        }

        // 食べ物の位置
        auto foods = stage.foods();
        file_out << foods[0].pos().id << " " << foods[1].pos().id << std::endl;
        
        // エージェントの着手(未実装)

        // 評価値(未実装)

    }

    // 最終順位の出力(未実装)
    file_out << "The output of the rankings is not yet available." << std::endl;
    return;
}

}