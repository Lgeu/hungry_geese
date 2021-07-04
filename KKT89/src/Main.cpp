#include "Simulator.hpp"

#include <cstdlib>
#include <cstring>

int main(int aArgc, const char* aArgv[]) {
    auto fSim = hungry_geese::Simulator();
    bool willPrintKif = false;

    if (aArgc > 1) {
        // 引数を記憶する
        for (int n = 1; n < aArgc; ++n) {
            // SEED値を指定
            if (std::strcmp(aArgv[n], "-r") == 0) {
                if (n + 1 < aArgc) {
                    uint x = uint(std::strtoul(aArgv[n + 1], nullptr, 0));
                    fSim.changeSeed(x);
                    n += 1;
                }
            }
            // 棋譜への出力を指定
            // 棋譜IDは日付にするように変更した
            else if (std::strcmp(aArgv[n], "-j") == 0) {
                willPrintKif = true;
                fSim.setKifID();
            }
        }
    }

    fSim.run();

    if (willPrintKif) {
        fSim.printKif();
    }
    
    return 0;
}