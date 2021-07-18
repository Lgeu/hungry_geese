#include "Simulator.hpp"

#include <cstdlib>
#include <cstring>
#include <chrono>

int main(int aArgc, const char* aArgv[]) {
    static auto fSim = hungry_geese::Simulator();
    bool willPrintKif = false;
    bool setSeed = false;

    if (aArgc > 1) {
        // 引数を記憶する
        for (int n = 1; n < aArgc; ++n) {
            // SEED値を指定
            if (std::strcmp(aArgv[n], "-r") == 0) {
                if (n + 1 < aArgc) {
                    uint x = uint(std::strtoul(aArgv[n + 1], nullptr, 0));
                    fSim.changeSeed(x);
                    setSeed = true;
                    n += 1;
                }
            }
            // 棋譜への出力を指定
            // 棋譜IDは日付にするように変更した
            else if (std::strcmp(aArgv[n], "-j") == 0) {
                willPrintKif = true;
            }
            // 探索時間の設定
            else if (std::strcmp(aArgv[n], "-t") == 0) {
                float timelimit = float(std::stof(aArgv[n + 1], nullptr));
                n += 1;
                fSim.SetTimeLimit(timelimit);
            }
            // パラメータの設定
            else if (std::strcmp(aArgv[n], "-p") == 0) {
                for (int i = 0; i < 4; ++i) {
                    fSim.parameter[i] = aArgv[n + 1 + i];
                }
                n += 4;
            }
            // 出力先のディレクトリを指定
            else if (std::strcmp(aArgv[n], "-o") == 0) {
                fSim.directory = aArgv[n + 1];
                n += 1;
            }
        }
    }

    // SEED値に指定がなかったらランダムに
    if (!setSeed) {
        fSim.changeSeed(std::chrono::steady_clock::now().time_since_epoch().count());
    }

    fSim.run();

    if (willPrintKif) {
        fSim.setKifID();
        fSim.printKif();
    }
    
    return 0;
}