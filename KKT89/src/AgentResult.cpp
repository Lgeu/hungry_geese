#pragma once
#include "AgentResult.hpp"

namespace hungry_geese {

AgentResult::AgentResult() : mAction(), mValue(-100), mAgentFeatures(), mConditionFeatures() {
    for (int i = 0; i < 4; ++i){
        mPolicy[i] = -100;
    }
}

}