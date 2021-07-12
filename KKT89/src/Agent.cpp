#pragma once
#include "Agent.hpp"

namespace hungry_geese {

Agent::Agent() : Agent0(), Agent1(), Agent2(), Agent3() {}

void Agent::setActions(Stage& aStage, int aIndex) {
    if (aIndex == 0) {
        auto result = Agent0.run(aStage, aIndex);
        aStage.mAgentResult[aIndex] = result;
        aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
    }
    else if (aIndex == 1) {
        auto result = Agent1.run(aStage, aIndex);
        aStage.mAgentResult[aIndex] = result;
        aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
    }
    else if (aIndex == 2) {
        auto result = Agent2.run(aStage, aIndex);
        aStage.mAgentResult[aIndex] = result;
        aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
    }
    else if (aIndex == 3) {
        auto result = Agent3.run(aStage, aIndex);
        aStage.mAgentResult[aIndex] = result;
        aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
    }
}

}