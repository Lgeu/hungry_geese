#include <iostream>
#include "./src/Parameter.hpp"
#include "./src/Stage.hpp"
#include "./src/AgentResult.hpp"
#include "./src/Duct.hpp"

void WriteAns(int direction) {
    if (direction == 0) {
        std::cout << "move u" << std::endl;
    }
    else if (direction == 1) {
        std::cout << "move r" << std::endl;
    }
    else if (direction == 2) {
        std::cout << "move d" << std::endl;
    }
    else {
        std::cout << "move l" << std::endl;
    }
    return;
}

int GetDirection(hungry_geese::Cpoint pre, hungry_geese::Cpoint cur) {
    int pre_x = pre.X(), pre_y = pre.Y();
    int cur_x = cur.X(), cur_y = cur.Y();
    if ((pre_x - 1 + hungry_geese::Parameter::rows) % hungry_geese::Parameter::rows == cur_x) return 0;
    else if ((pre_y + 1) % hungry_geese::Parameter::columns == cur_y) return 1;
    else if ((pre_x + 1) % hungry_geese::Parameter::rows == cur_x) return 2;
    else return 3;
}

void Input(std::array<hungry_geese::Stage, 200> &stages, const int idx) {
    auto stage = stages[idx];
    std::cin >> stage.mRemainingTime[0];
    std::cin >> stage.mTurn;
    for (int i = 0; i < 4; ++i) {
        int n; std::cin >> n;
        for (int j = 0; j < n; ++j) {
            int g; std::cin >> g;
            stage.mGeese[i].mItems.push(hungry_geese::Cpoint(g));
        }
    }
    for (int i = 0; i < 2; ++i) {
        int g; std::cin >> g;
        stage.mFoods[i] = hungry_geese::Cpoint(g);
    }
    for (int i = 0; i < 4; ++i) {
        stage.mLastActions[i] = hungry_geese::Action(0);
    }
    if (idx > 0) {
        for (int i = 0; i < 4; ++i) {
            if (stage.mGeese[i].mItems.size() > 0) {
                stage.mLastActions[i] = hungry_geese::Action(GetDirection(stages[idx - 1].mGeese[i].mItems[0], stages[idx].mGeese[i].mItems[0]));
            }
        }
    }
}

int main(int aArgc, const char* aArgv[]) {

    static std::array<hungry_geese::Stage, 200> stages;
    static hungry_geese::Duct Agent;
    float timelimit = 0.3;

    if (aArgc > 1) {
        // 引数を記憶する
        for (int n = 1; n < aArgc; ++n) {
            // 探索時間の設定
            if (std::strcmp(aArgv[n], "-t") == 0) {
                timelimit = float(std::stof(aArgv[n + 1], nullptr));
                n += 1;
            }
            // パラメータの設定
            else if (std::strcmp(aArgv[n], "-p") == 0) {
                Agent.nnue.SetParameter(aArgv[n + 1]);
                n += 1;
            }
        }
    }

    // ゲーム進行
    for (int turn = 0; turn < 200; ++turn) {
        // input
        Input(stages, turn);
        // search
        Agent.InitDuct(stages[turn], 0);
        hungry_geese::AgentResult res = Agent.Search(timelimit);
        // output
        WriteAns(res.mAction);
    }

    return 0;
}