#pragma once
#include "Parameter.hpp"
#include "Stack.hpp"
#include "Point.hpp"

namespace hungry_geese {

using Items = Stack<Point, 77>;

struct Goose {
	// コンストラクタ
	Goose();
	Goose(Point aPos);

	// 問い合わせ
	// 占めているグリッド
	// 配列の末尾を尻尾にしたい
	Items items() const;
	// 生き残ってるかどうか
	bool isSurvive() const;
	// 生き残ってるかどうかを設定する
	void setIsSurvive(bool isSurvive);
	// 残り時間
	float remainingTime() const;

	// メンバ変数
	// Goose が占めているグリッド
	Items mItems;
	// 生き残ってるかどうか
	bool mIsSurvive;
	// 残り時間
	float mRemainingTime;
};

}