#pragma once
#include <cstring>
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "AgentResult.hpp"
#include "Stage.hpp"

namespace hungry_geese {

namespace evaluation_function {
namespace feature {
void PrintFeatureBoundary();
}
namespace test {
void TestModel();
void TestEvaluator();
}
}

struct GreedyAgent {
	// コンストラクタ
	GreedyAgent();

	// 実行
	AgentResult run(const Stage& aStage, int aIndex);

	// 関数
	int min_Distance(Point aPos, Point bPos);
	Point Translate(Point aPos, int Direction);

	// いつもの
	const std::array<int, 4> dx = {-1, 0, 1, 0};
	const std::array<int, 4> dy = {0, 1, 0, -1};
	const int INF = 1e9;
};

}