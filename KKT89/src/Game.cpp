#pragma once
#include "Game.hpp"

namespace hungry_geese {

Game::Game() : mTurn(0), mStages() {}

int Game::turn() const {
	return mTurn;
}

}

// かなり未実装