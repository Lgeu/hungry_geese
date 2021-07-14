#pragma once
#include "AgentResult.hpp"
#include "Stage.hpp"
#include "Stack.hpp"
#include "Assert.hpp"
#include "Evaluation_function.hpp"

namespace hungry_geese {

struct MctsAgent {
    // コンストラクタ
    MctsAgent();

    constexpr static int BIG = 100000;

    // Node_type
    enum struct NodeType : bool {
        AGENT_NODE,
        FOOD_NODE,
    };

    // State
    struct State {
        State();
        State(hungry_geese::Stage aStage, int aIndex);
        std::array<Stack<Point, 77>, 4> geese;
        std::array<Point, 2> foods;
        int current_step;
        std::array<int, 4> last_actions;

        void debug();
        // 手を引数で渡して次のノードを得る
        State NextState(NodeType node_type, const std::array<int, 4> agent_action) const;
        // シミュレート
        static void Simulate(State &state, const std::array<int ,4>& agent_action);
    };

    // Node
    struct Node {
        State state; // 状態
        std::array<std::array<float, 4>, 4> policy; // 各エージェントの方策
        std::array<std::array<float, 4>, 4> worth; // 各エージェント視点の各手の価値
        int n; // このノードで行動を起こした回数 (=到達回数 - 1)
        int n_children; // 子ノードの数
        int children_offset; // 子ノード
        NodeType node_type; // エージェントのノード (0) か、食べ物のノード (1) か

        // 空きマスをstackで管理する
        Stack<int, 77> empty_cell;

        Node();

        Node(const State& aState, Stack<Node*, BIG>& children_buffer);

        bool Expanded() const; // 既にモデルによって評価されているかを返す
        // 例の式を使って (食べ物のノードの場合はランダムに) 手を選ぶ  // 手の全パターンをひとつの値で表す。全員が 3 方向に移動できるなら 0 から 80 までの値をとる。Move メソッドで具体的な手に変換できる
        int ChooseMove();
        // InitCell()が呼び出されたかどうか
        bool do_empty_cell() const;
        // empty_cellを求める
        void InitCell();
        // idx_move に対応する idx_agent 番目のエージェントの手を返す
        int Move(const int& idx_move, const int& idx_agent);
        // k 番目の行動によって遷移する子ノードを返す 
        // その子ノードが初めて遷移するものだった場合、新たに領域を確保してノードを作る
        Node& KthChildren(Stack<Node, BIG>& node_buffer, Stack<Node*, BIG>& children_buffer, const int& k);
    };

    Stack<Node, BIG> node_buffer;
    Stack<Node*, BIG> children_buffer;
    std::array<int, BIG> move_buffer;

    // Node用のメモリを予め確保しておく
    // static constexpr int NodeSize = 1000;
    // std::array<Node, NodeSize> mNodes;
    // int mNodeIndex = 0; // 使い回しの為のカウンター
    // bool push(Node aNode); // mNodeにpushする

    // 実行
    AgentResult run(const Stage& aStage, int aIndex);

    // 評価値最大の行動を返す
    AgentResult solve1(const Stage& aStage, int aIndex);

    // 評価
    Evaluator mEvaluator;

    // 方向を指定して移動先を返す関数
    static Point Translate(Point aPos, int Direction);

    // いつもの
    static constexpr std::array<int, 4> dx = {-1, 0, 1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};

    // 行動一覧
    static constexpr Actions Idx_to_Actions = {
        Action::NORTH,
        Action::EAST,
        Action::SOUTH,
        Action::WEST,
    };
};

}