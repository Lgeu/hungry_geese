#pragma once
#include "Game.hpp"

namespace hungry_geese {

Game::Game() : mTurn(0), mStages() {
	for (int i = 0; i < 4; ++i) {
		mRewards[i] = 0;
		mStanding[i] = 1;
	}
	for (int i = 0; i < 200; ++i) {
		mStages[i].mTurn = i;
	}
}

int Game::turn() const {
	return mTurn;
}

void Game::calc_Ranking() {
	int score[4];
	for (int i = 0; i < 4; ++i) {
		score[i] = mRewards[i];
	}
	std::sort(score, score+4);
	for (int i = 0; i < 4; ++i) {
		for (int j = 3; j >= 0; --j) {
			if (score[j] == mRewards[i]) {
				mStanding[i] = 4 - j;
				break;
			}
		}
	}
}

}