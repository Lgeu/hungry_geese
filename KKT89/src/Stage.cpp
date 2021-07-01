#pragma once
#include "Stage.hpp"

namespace hungry_geese {

Stage::Stage() : mGeese(), mFoods(), mBoard() {}

const Geese& Stage::geese() const {
	return mGeese;
}

const Foods& Stage::foods() const {
	return mFoods;
}

bool Stage::isEnd() const {
	int Survivor = 0;
	for (Goose goose: mGeese) {
		if (goose.isSurvive()) {
			++Survivor;
		}
	}
	return (Survivor == 1);
}

void Stage::InitializeBoard() {
	// 初期化
	for (int i = 0; i < 77; ++i) {
		mBoard[i] = 0;
	}

	// Goose
	for (Goose goose: geese()) {
		auto items = goose.items();
		for (int i = 0; i < items.right; ++i) {
			auto pos = items[i];
			mBoard[pos.id] = 1;
		}
	}
	// Food
	for (Food food: foods()) {
		auto pos = food.pos();
		mBoard[pos.id] = 1;
	}

	return;
}

bool Stage::isEmpty(int aId) const {
	return (mBoard[aId] == 0);
}

}