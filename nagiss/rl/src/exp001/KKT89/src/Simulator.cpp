#pragma once
#define _CRT_SECURE_NO_WARNINGS 
#include "Simulator.hpp"

namespace hungry_geese {
    
Simulator::Simulator() : mGame(), mAgent(), mTimer() {
    rand = Random(0);
}

void Simulator::changeSeed(uint aX) {
    rand = Random(aX);
}

std::string Simulator::getDatetimeStr() const {
    time_t t = time(nullptr);
    const tm* localTime = localtime(&t);
    std::stringstream s;
    s << "20" << localTime->tm_year - 100;
    // setw(),setfill()で0詰め
    s << std::setw(2) << std::setfill('0') << localTime->tm_mon + 1;
    s << std::setw(2) << std::setfill('0') << localTime->tm_mday;
    s << std::setw(2) << std::setfill('0') << localTime->tm_hour;
    s << std::setw(2) << std::setfill('0') << localTime->tm_min;
    s << std::setw(2) << std::setfill('0') << localTime->tm_sec;
    // std::stringにして値を返す
    return s.str();
}

void Simulator::setKifID() {
    std::string Date = getDatetimeStr();
    std::string Seed = std::to_string(rand.x);
    KifID = Date + "_" + Seed;
}

void Simulator::run() {
    // 初期条件の設定
    auto &stage0 = mGame.mStages[0];
    stage0.InitializeBoard();
    // Goose
    for (int i = 0; i < 4; ++i) {
        int id = rand.randTerm(77);
        id = stage0.randPos(id);
        stage0.mGeese[i] = Goose(Point(id));
        stage0.mBoard[id] = 1;
    }
    // 食べ物
    for (int i = 0; i < 2; ++i) {
        int id = rand.randTerm(77);
        id = stage0.randPos(id);
        stage0.mFoods[i] = Food(Point(id));
        stage0.mBoard[id] = 1;
    }
    // 残り時間
    for (int i = 0; i < 4; ++i) {
        stage0.mRemainingTime[i] = Parameter::remainingOverageTime;
    }

    // ゲーム終了までターンを進行させる
    for (int i = 0; i+1 < Parameter::episodeSteps; ++i) {
        auto &stage = mGame.mStages[i];
        auto &nextstage = mGame.mStages[i+1];

        // 終了判定
        if (stage.isEnd()) {
            for (int j = 0; j < 4; ++j) {
                stage.mGeese[j].setIsSurvive(false);
            }
            break;
        }

        // デバック
        if (false) {
            std::cerr << i << std::endl;
            std::array<char, 77> debug;
            for (int j = 0; j < 77; ++j) {
                debug[j] = '*';
            }
            // Goose
            for (Goose goose: stage.geese()) {
                auto items = goose.items();
                for (int i = 0; i < items.right; ++i) {
                    auto pos = items[i];
                    debug[pos.id] = 'S';
                    if (i == 0) {
                        debug[pos.id] = 'A';
                    }
                    else if (i + 1 == items.right) {
                        debug[pos.id] = 'B';
                    }
                }
            }
            // 食べ物
            debug[stage.mFoods[0].pos().id] = 'F';
            debug[stage.mFoods[1].pos().id] = 'F';
            for (int r = 0; r < Parameter::rows; ++r) {
                for (int c = 0; c < Parameter::columns; ++c) {
                    std::cerr << debug[r*Parameter::columns + c];
                }
                std::cerr << std::endl;
            }
        }

        // 探索
        for (int j = 0; j < 4; ++j){
            if (!stage.geese()[j].isSurvive()) {
                continue;
            }

            mTimer.start();
            mAgent.setActions(stage, j);
            mTimer.stop();

            // 残り時間の計算
            if (i > 0) {
                stage.mRemainingTime[j] = mGame.mStages[i-1].mRemainingTime[j];
            }
            float get_time = mTimer.elapsedSec();
            get_time = 1.0 - get_time;
            if (get_time < 0) {
                stage.mRemainingTime[j] -= get_time;
            }
        }

        // 着手
        auto geese = stage.geese();
        auto act = stage.actions();
        for (int j = 0; j < 4; ++j) {
            auto goose = geese[j];

            // 既に脱落している場合
            if (!goose.isSurvive()) {
                nextstage.mGeese[j].setIsSurvive(false);
                continue;
            }

            // 命令に従って頭の移動する位置を求める
            // 前ターンの反対の動きをしていないか判定
            auto items = goose.items();
            auto head = items[0];
            int x = head.x, y = head.y;
            if (act[j] == Action::NORTH) {
                nextstage.mLastActions[j] = Action::NORTH;
                if (stage.mLastActions[j] == Action::SOUTH) {
                    nextstage.mGeese[j].setIsSurvive(false);
                    continue;
                }
                --x;
                if (x < 0) {
                    x += Parameter::rows;
                }
            }
            else if (act[j] == Action::EAST) {
                nextstage.mLastActions[j] = Action::EAST;
                if (stage.mLastActions[j] == Action::WEST) {
                    nextstage.mGeese[j].setIsSurvive(false);
                    continue;
                }
                ++y;
                if (y >= Parameter::columns) {
                    y = 0;
                }
            }
            else if (act[j] == Action::SOUTH) {
                nextstage.mLastActions[j] = Action::SOUTH;
                if (stage.mLastActions[j] == Action::NORTH) {
                    nextstage.mGeese[j].setIsSurvive(false);
                    continue;
                }
                ++x;
                if (x >= Parameter::rows) {
                    x = 0;
                }
            }
            else if (act[j] == Action::WEST) {
                nextstage.mLastActions[j] = Action::WEST;
                if (stage.mLastActions[j] == Action::EAST) {
                    nextstage.mGeese[j].setIsSurvive(false);
                    continue;
                }
                --y;
                if (y < 0) {
                    y += Parameter::columns;
                }
            }

            // 次のターンの頭の位置
            Point nextHead = Point(x,y);

            // 食べ物を食べたかどうか
            bool eatFood = false;
            if (stage.mFoods[0].pos() == nextHead and !stage.mFoods[0].isEaten()) {
                eatFood = true;
                stage.mFoods[0].setIsEaten(true);
            }
            if (stage.mFoods[1].pos() == nextHead and !stage.mFoods[1].isEaten()) {
                eatFood = true;
                stage.mFoods[1].setIsEaten(true);
            }
            if (!eatFood) {
                items.pop();
            }

            // 自己衝突判定
            bool selfCollision = false;
            for (int k = 0; k < items.right; ++k) {
                if (items[k] == nextHead) {
                    selfCollision = true;
                    break;
                }
            }
            if (selfCollision) {
                nextstage.mGeese[j].setIsSurvive(false);
                continue;
            }

            // 次のターンのgooseのstackを用意する
            nextstage.mGeese[j] = Goose(nextHead);
            for (int k = 0; k < items.right; ++k) {
                nextstage.mGeese[j].mItems.push(items[k]);
            }

            // hunger_rateの考慮
            if ((i + 1) % Parameter::hunger_rate == 0) {
                if (nextstage.mGeese[j].mItems.size() > 0) {
                    nextstage.mGeese[j].mItems.pop();
                }
                if (nextstage.mGeese[j].mItems.size() == 0) {
                    nextstage.mGeese[j].setIsSurvive(false);
                    continue;
                }
            }
        }

        // 2体以上の衝突判定
        std::array<int, 77> goose_positions;
        for (int k = 0; k < 77; ++k) {
            goose_positions[k] = 0;
        }
        for (Goose goose: nextstage.geese()) {
            if (!goose.isSurvive()) continue;
            auto items = goose.items();
            for (int j = 0; j < items.right; ++j) {
                auto pos = items[j];
                goose_positions[pos.id]++;
            }
        }
        for (int j = 0; j < 4; ++j) {
            Goose goose = nextstage.mGeese[j];
            if (!goose.isSurvive()) continue;
            auto head = goose.items()[0];
            if (goose_positions[head.id] > 1) {
                nextstage.mGeese[j].setIsSurvive(false);
                continue;
            }
        }

        // 食べ物の補充
        int needed_food = 0;
        for (Goose goose: nextstage.geese()) {
            if (!goose.isSurvive()) continue;
            auto items = goose.items();
            for (int j = 0; j < items.right; ++j) {
                auto pos = items[j];
                nextstage.mBoard[pos.id] = 1;
            }
        }
        if (!stage.mFoods[0].isEaten()) {
            nextstage.mFoods[0] = stage.mFoods[0];
            nextstage.mBoard[nextstage.mFoods[0].pos().id] = 1;
        }
        else {
            ++needed_food;
        }
        if (!stage.mFoods[1].isEaten()) {
            nextstage.mFoods[1] = stage.mFoods[1];
            nextstage.mBoard[nextstage.mFoods[1].pos().id] = 1;
        }
        else {
            ++needed_food;
        }
        Stack<int, 77> available_positions;
        for (int j = 0; j < 77; ++j) {
            if (nextstage.mBoard[j] == 0) {
                available_positions.push(j);
            }
        }
        if (available_positions.size() < needed_food) {
            needed_food = available_positions.size();
        }
        int used_food = -1;
        if (stage.mFoods[0].isEaten() and needed_food > 0) {
            int id = rand.randTerm(available_positions.size());
            id = available_positions[id];
            nextstage.mFoods[0] = Food(Point(id));
            nextstage.mBoard[nextstage.mFoods[0].pos().id] = 1;
            --needed_food;
            used_food = id;
        }
        if (stage.mFoods[1].isEaten() and needed_food > 0) {
            int id = rand.randTerm(available_positions.size());
            id = available_positions[id];
            while (id == used_food) {
                id = rand.randTerm(available_positions.size());
                id = available_positions[id];
            }
            nextstage.mFoods[1] = Food(Point(id));
            nextstage.mBoard[nextstage.mFoods[1].pos().id] = 1;
            --needed_food;
        }

        // Rewardの算出
        for (int j = 0; j < 4; ++j) {
            Goose goose = nextstage.mGeese[j];
            if (!goose.isSurvive()) continue;
            mGame.mRewards[j] = (i + 1) * (Parameter::max_length + 1) + goose.items().size();
        }

        // ターン数を進める
        ++mGame.mTurn;
    }

    // 順位確定
    mGame.calc_Ranking();

    return;
}

void Simulator::printKif() const {
    std::string filename = "../src/out/" + KifID + KifExtension;
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

        // エージェントの残り時間
        for (int i = 0; i < 4; ++i) {
            if (i > 0) {
                file_out << " ";
            }
            if (!stage.geese()[i].isSurvive()) {
                file_out << -100;
            }
            else {
                file_out << stage.mRemainingTime[i];
            }
        }
        file_out << std::endl;

        // エージェントの位置
        for (Goose goose: stage.geese()) {
            auto items = goose.items();
            int size = items.right;
            if (!goose.isSurvive()) {
                size = 0;
            }
            file_out << size;
            for (int i = 0; i < size; ++i) {
                auto pos = items[i];
                file_out << " " << pos.id;
            }
            file_out << std::endl;
        }

        // 食べ物の位置
        auto foods = stage.foods();
        file_out << foods[0].pos().id << " " << foods[1].pos().id << std::endl;

        // エージェントの着手
        auto act = stage.actions();
        for (int j = 0; j < 4; ++j) {
            auto goose = stage.geese()[j];
            if (j > 0) {
                file_out << " ";
            }
            // 既に脱落している場合、-100
            if (!goose.isSurvive()) {
                file_out << -100;
                continue;
            }
            if (act[j] == Action::NORTH) {
                file_out << 0;
            }
            else if (act[j] == Action::EAST) {
                file_out << 1;
            }
            else if (act[j] == Action::SOUTH) {
                file_out << 2;
            }
            else if (act[j] == Action::WEST) {
                file_out << 3;
            }
        }
        file_out << std::endl;

        // 評価値
        for (int i = 0; i < 4; ++i){
            auto AgentResult = stage.mAgentResult[i];
            // 盤面評価値
            file_out << AgentResult.mValue;
            for (int j = 0; j < 4; ++j) {
                file_out << " " << AgentResult.mPolicy[j];
            }
            file_out << std::endl;
        }

        // 特徴量ベクトル
        for (int currentlySurvivingAgent = 0; currentlySurvivingAgent < 4; currentlySurvivingAgent++) {
            if (!stage.geese()[currentlySurvivingAgent].isSurvive()) continue;  // 4 エージェントとも同じ情報を持っている (はず) ので、生きてるやつを使う
            const auto& agentResult = stage.mAgentResult[currentlySurvivingAgent];
            // 各エージェントの特徴
            for (int i = 0; i < 4; i++) {
                file_out << agentResult.mAgentFeatures[i].size();
                for (const auto& feature : agentResult.mAgentFeatures[i]) {
                    file_out << " " << feature;
                }
                file_out << std::endl;
            }
            // 全体の特徴
            file_out << agentResult.mConditionFeatures.size();
            for (const auto& feature : agentResult.mConditionFeatures) {
                file_out << " " << feature;
            }
            file_out << std::endl;
            goto end_print_feature;
        }
        // 生きてるやつがいなかった場合の処理
        for (int i = 0; i < 5; i++) file_out << 0 << std::endl;

    end_print_feature:;


    }

    // 最終順位の出力
    auto standing = mGame.mStanding;
    file_out << standing[0] << " " << standing[1] << " " << standing[2] << " " << standing[3] << std::endl;

    return;
}

}