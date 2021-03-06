#pragma once
#include <math.h>
#include <cmath>
#include <random>
#include <chrono>
#include "Point.hpp"
#include "Stage.hpp"
#include "AgentResult.hpp"
#include "library.hpp"
#include "Evaluator2.hpp"

namespace hungry_geese {

struct Duct {
    // コンストラクタ
    Duct();

    constexpr static int node_buffer_size = 400000;
    constexpr static int children_buffer_size = 20000000;
    void Setprintlog(bool f);
    bool printlog = false;

    // NodeType
    enum struct NodeType : bool {
        AGENT_NODE,
        FOOD_NODE,
    };

    // State
    struct State {
        State();
        State(hungry_geese::Stage aStage, int aIndex);
        std::array<Cpoint,77> geese;
        std::array<signed char, 5> boundary;
        std::array<Cpoint, 2> foods;
        unsigned char current_step; // ターン数
        unsigned char last_actions; // [0,256)の整数で手を表現する
        std::array<signed char, 4> ranking;

        // idx_agent 番目のgooseのサイズを返す
        signed char goose_size(signed char idx_agent);
        // 手を引数で渡して次のノードを得る
        // food_sub : 食べ物が2個同時に消える場合256だと情報足りない
        State NextState(NodeType node_type, const unsigned char agent_action, const unsigned char food_sub) const;
        // シミュレート
        static void Simulate(State &state, const unsigned char& agent_action);
        // 終局状態か(プレイヤー0が生存しているか)
        bool Finished() const;
        // デバック用
        void Debug() const; 
    };

    // Node
    struct Node {
        State state; // 状態
        std::array<std::array<float, 4>, 4> policy; // 各エージェントの方策
        std::array<float, 4> value; // 状態評価値4人分
        std::array<std::array<float, 4>, 4> worth; // 各エージェント視点の各手の累積価値
        std::array<std::array<int, 4>, 4> n; // 各手の選ばれた回数
        int visited; // このノードを訪れた回数
        int n_children; // 子ノードの数(実際の子供の数) // 食べ物の遷移先頂点数はmin(n_children, 小さい定数)
        int children_offset; // 子ノード
        NodeType node_type; // エージェントのノード (0) か、食べ物のノード (1) か

        // 問い合わせ
        const std::array<std::array<float, 4>, 4>& GetPolicy() const;
        const std::array<std::array<float, 4>, 4>& GetWorth() const;

        Node();

        Node(const State& aState, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer);

        int SmallChildrenSize() const;
        bool Expanded() const; // 既にモデルによって評価されているかを返す
        // アーク評価値
        float Argvalue(const int& idx_agent, const int& idx_move, const int& t_sum);
        // 例の式を使って (食べ物のノードの場合はランダムに) 手を選ぶ  // 手の全パターンをひとつの値で表す。全員が 3 方向に移動できるなら 0 から 80 までの値をとる。Move メソッドで具体的な手に変換できる
        // 7/24修正：返り値が{k番目の子, k番目のposition_id}に
        // pairは嫌なのでKthChildrenで計算することにした
        int ChooseMove(const int& t_sum, const int& cnt);
        // k 番目の行動によって遷移する子ノードを返す 
        // その子ノードが初めて遷移するものだった場合、新たに領域を確保してノードを作る
        Node& KthChildren(nagiss_library::Stack<Node, node_buffer_size>& node_buffer, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer, const int& k, nagiss_library::PermutationGenerator generator);
        // デバック用
        void Debug() const;
    };

    nagiss_library::Stack<Node, node_buffer_size> node_buffer;
    nagiss_library::Stack<Node*, children_buffer_size> children_buffer;
    evaluation_function::Evaluator nnue;
    nagiss_library::PermutationGenerator p_generator;
    int t_sum; // 累計試行回数

    void InitDuct(hungry_geese::Stage aStage, int aIndex);

    // 探索
    // 返り値をAgentResultにしたい
    AgentResult Search(const float timelimit);
    Node& RootNode();
    void Iterate();

    // 方向を指定して移動先を返す関数
    static Cpoint Translate(Cpoint aPos, int Direction);

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