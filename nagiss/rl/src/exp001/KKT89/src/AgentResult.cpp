#pragma once
#include "AgentResult.hpp"

namespace hungry_geese {

AgentResult::AgentResult() : mAction(), mValue(-100) {
	for (int i = 0; i < 4; ++i){
		mPolicy[i] = -100;
	}
	// TODO：特徴量の初期化について
}

}