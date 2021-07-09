#pragma once
#include <array>
#include "Food.hpp"
#include "Goose.hpp"
#include "Parameter.hpp"
#include "Action.hpp"

namespace hungry_geese {

// Gooseの配列
using Geese = std::array<Goose, 4>;
// 食べ物の配列
using Foods = std::array<Food, Parameter::min_food>;
// 盤面の配列
using Board = std::array<int, 77>;
// 行動の配列
using Actions = std::array<Action, 4>;

struct Stage {
	// コンストラクタ
	Stage();

	// Goose*4の配列
    const Geese& geese() const;
    // 食べ物の配列
    const Foods& foods() const;
    // 行動の配列
    const Actions& actions() const;

    // ゲーム終了しているか
    bool isEnd() const;
    // 盤面の初期化
    void InitializeBoard();
    // そのマスが空かどうか(先に初期化をする)
    bool isEmpty(int aId) const;
    // ランダムに一点持ってきて一番近い空マスを返す
    // 盤面全部埋まってた時は aId-1(mod77) を返す
    int randPos(int aId) const;

	// メンバ変数
    // 現在のターン数
    int mTurn;
    // Gooseの配列
    Geese mGeese;
    // 食べ物の配列
    Foods mFoods;
    // 盤面の配列
    Board mBoard;
    // 行動の配列
    Actions mActions;
    // 前ターンの行動の配列
    Actions mLastActions;
};

}