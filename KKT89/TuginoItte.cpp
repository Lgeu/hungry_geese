#include <iostream>
#include "./src/Parameter.hpp"
#include "./src/Stage.hpp"
#include "./src/AgentResult.hpp"
#include "./src/Duct.hpp"
#include "./src/Duct2.hpp"

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

void Input(hungry_geese::Stage &stage) {
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
        int g; std::cin >> g;
        stage.mLastActions[i] = hungry_geese::Action(g);
    }
}

int main(int aArgc, const char* aArgv[]) {

    static hungry_geese::Stage stage;
    static hungry_geese::Duct Agent;
    static hungry_geese::Duct2 Agent2;
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

    // input
    Input(stage);

    // 次の一手(4人分)
    for (int i = 0; i < 4; ++i) {
        if (stage.mGeese[i].mItems.size() == 0) {
            std::cout << "Agent dropped out." << std::endl;
        }
        else {
            // search
            Agent2.InitDuct(stage, i);
            hungry_geese::AgentResult res = Agent2.Search(timelimit);
            // output
            WriteAns(res.mAction);
            // auto rootnode = Agent.RootNode();
            // for (int j = 0; j < 4; ++j) {
            //     std::cout << rootnode.policy[i][j] << " ";
            // }
            // std::cout << std::endl;
        }
    }

    return 0;
}