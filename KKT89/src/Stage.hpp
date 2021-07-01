#pragma once
#include <array>
#include "Food.hpp"
#include "Goose.hpp"
#include "Parameter.hpp"

namespace hungry_geese {

// Gooseの配列
using Geese = std::array<Goose, 4>;
// 食べ物の配列
using Foods = std::array<Food, Parameter::min_food>;
// 盤面の配列
using Board = std::array<int, 77>;

struct Stage {
	// コンストラクタ
	Stage();

	// Goose*4の配列
    const Geese& geese() const;
    // 食べ物の配列
    const Foods& foods() const;

    // ゲーム終了しているか
    bool isEnd() const;
    // 盤面の初期化
    void InitializeBoard();
    // そのマスが空かどうか
    bool isEmpty(int aId) const;

	// メンバ変数
    // Gooseの配列
    Geese mGeese;
    // 食べ物の配列
    Foods mFoods;
    // 盤面の配列
    Board mBoard;
};

}

// エージェントの行動の配列
// エージェントの評価値