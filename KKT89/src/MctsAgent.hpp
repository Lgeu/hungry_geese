#pragma once
#include "AgentResult.hpp"
#include "Stage.hpp"
#include "Stack.hpp"
#include "Evaluation_function.hpp"

namespace hungry_geese {

struct MctsAgent {
	// コンストラクタ
	MctsAgent();

	// Node
    struct Node {
        Stage state; // 状態
        float w; // 累計価値
        int n; // 試行回数
        bool is_AgentTurn; // Agentの手番か
        Stack<int, 81> child_nodes; // 子ノードのindexを持つ
    };
    // Node用のメモリを予め確保しておく
    const int NodeSize = 1000;
    // std::array<Node, NodeSize> みたいに書きたかったけど書けなかった
    std::array<Node, 100> mNodes;

	// 実行
	AgentResult run(const Stage& aStage, int aIndex);

	// 評価
	Evaluator mEvaluator;

	// 関数
	int min_Distance(Point aPos, Point bPos);
	Point Translate(Point aPos, int Direction);

	// いつもの
	const std::array<int, 4> dx = {-1, 0, 1, 0};
	const std::array<int, 4> dy = {0, 1, 0, -1};
	const int INF = 1e9;
};

}