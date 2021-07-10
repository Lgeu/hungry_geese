#pragma once
#include "Agent.hpp"

namespace hungry_geese {

Agent::Agent() {}

void Agent::setActions(Stage& aStage, int aIndex) {
	if (aIndex == 0) {
		auto result = Agent0::Agent.run(aStage, aIndex);
		aStage.mAgentResult[aIndex] = result;
		aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
	}
	else if (aIndex == 1) {
		auto result = Agent1::Agent.run(aStage, aIndex);
		aStage.mAgentResult[aIndex] = result;
		aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
	}
	else if (aIndex == 2) {
		auto result = Agent2::Agent.run(aStage, aIndex);
		aStage.mAgentResult[aIndex] = result;
		aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
	}
	else if (aIndex == 3) {
		auto result = Agent3::Agent.run(aStage, aIndex);
		aStage.mAgentResult[aIndex] = result;
		aStage.mActions[aIndex] = Idx_to_Actions[result.mAction];
	}
}

}