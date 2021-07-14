#pragma once
#include "MctsAgent.hpp"

namespace hungry_geese {

//------------------------------------------------------------------------------
// コンストラクタ
MctsAgent::MctsAgent() : mEvaluator() {}

//------------------------------------------------------------------------------
// State
MctsAgent::State::State() : geese(), foods(), current_step(), last_actions() {}

MctsAgent::State::State(hungry_geese::Stage aStage, int aIndex) {
    for (int i = 0; i < 4; ++i) {
        if (!aStage.geese()[i].isSurvive()) {
            continue;
        }
        geese[i] = aStage.geese()[i].items();
    }
    for (int i = 0; i < 2; ++i) {
        foods[i] = aStage.foods()[i].pos();
    }
    std::swap(geese[0], geese[aIndex]);
    current_step = aStage.mTurn;
    auto LastActions = aStage.mLastActions;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (LastActions[i] == Idx_to_Actions[j]) {
                last_actions[i] = j;
            }
        }
    }
    std::swap(last_actions[0], last_actions[aIndex]);
}

void MctsAgent::State::debug() {
    std::cerr << current_step << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cerr << geese[i].size();
        for (int j = 0; j < geese[i].size(); ++j) {
            std::cerr << " " << geese[i][j].id;
        }
        std::cerr << std::endl;
    }
    std::cerr << foods[0].id << " " << foods[1].id << std::endl;
    std::cerr << last_actions[0] << " " << last_actions[1] << " " << last_actions[2] << " " << last_actions[3] << std::endl;
}

MctsAgent::State MctsAgent::State::NextState(NodeType node_type, const std::array<int, 4> agent_action) const {
    State nextstate;
    nextstate.geese = geese;
    nextstate.foods = foods;
    nextstate.current_step = current_step;
    nextstate.last_actions = last_actions;
    if (node_type == NodeType::AGENT_NODE) {
        Simulate(nextstate, agent_action);
        for (int i = 0; i < 4; ++i) {
            nextstate.last_actions[i] = agent_action[i];
        }
        nextstate.current_step++;
    }
    else {
        for (int i = 0; i < 2; ++i) {
            if (foods[i].id == -1) {
                nextstate.foods[i].id = agent_action[0];
                break;
            }
        }
    }
    return nextstate;
}

void MctsAgent::State::Simulate(State &state, const std::array<int ,4>& agent_action) {
    for (int i = 0; i < 4; ++i) {
        if (state.geese[i].size() == 0) {
            continue;
        }
        auto head = MctsAgent::Translate(state.geese[i][0], agent_action[i]);
        bool eatFood = false;
        for (int j = 0; j < 2; ++j) {
            if (head == state.foods[j]) {
                eatFood = true;
                state.foods[j] = hungry_geese::Point(-1);
            }
        }
        if (!eatFood) {
            state.geese[i].pop();
        }
        for (int j = 0; j < state.geese[i].size(); ++j) {
            if (head == state.geese[i][j]) {
                state.geese[i].clear();
                break;
            }
        }
        if (state.geese[i].size()) {
            auto goose = state.geese[i];
            state.geese[i].clear();
            state.geese[i].push(head);
            for (int j = 0; j < goose.size(); ++j) {
                state.geese[i].push(goose[j]);
            }
        }
        if ((state.current_step + 1) % hungry_geese::Parameter::hunger_rate == 0) {
            if (state.geese[i].size() > 0) {
                state.geese[i].pop();
            }
        }
    }
    static std::array<int, 77> simulate_goose_positions; // シミュレーション用の配列
    for (int i = 0; i < 77; ++i) {
        simulate_goose_positions[i] = 0;
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < state.geese[i].size(); ++j) {
            simulate_goose_positions[state.geese[i][j].id]++;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (state.geese[i].size() > 0) {
            auto head = state.geese[i][0];
            if (simulate_goose_positions[head.id] > 1) {
                state.geese[i].clear();
            }
        }
    }
}

//------------------------------------------------------------------------------
// Node
MctsAgent::Node::Node() : state(), policy(), worth(), n(0), n_children(), children_offset(), node_type(), empty_cell() {}

MctsAgent::Node::Node(const State& aState, Stack<Node*, BIG>& children_buffer) : state(aState), policy(), worth(), n(0), children_offset(), node_type(), empty_cell() {
    policy[0][0] = -100.0;

    if (aState.foods[0].id == -1 or aState.foods[1].id == -1) {
        node_type = NodeType::FOOD_NODE;
    }

    // 子ノードの数を数える処理
    if (node_type == NodeType::AGENT_NODE) {
        n_children = 1;
        for (int i = 0; i < 4; ++i) {
            if (aState.geese[i].size() > 0) {
                n_children *= 3;
            }
        }
    }
    else {
        n_children = 77;
        for (int i = 0; i < 2; ++i) {
            if (aState.foods[i].id != -1) {
                n_children--;
            }
        }
        for (int i = 0; i < 4; ++i) {
            n_children -= aState.geese[i].size();
        }
    }

    children_offset = children_buffer.size();
    children_buffer.resize(children_offset + n_children);
}

bool MctsAgent::Node::Expanded() const {
    return (policy[0][0] != -100.0);
}

int MctsAgent::Node::ChooseMove() {
    // 未実装
    return 0;
}

bool MctsAgent::Node::do_empty_cell() const {
    if (n_children > 0 and empty_cell.size() == 0) {
        return false;
    }
    else {
        return true;
    }
}

void MctsAgent::Node::InitCell() {
    static std::array<bool, 77> used;
    for (int i = 0; i < 77; ++i) {
        used[i] = false;
    }
    auto geese = state.geese;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < geese[i].size(); ++j) {
            used[geese[i][j].id] = true;
        }
    }
    auto foods = state.foods;
    for (int i = 0; i < 2; ++i) {
        if (foods[i].id != -1) {
            used[foods[i].id] = true;
        }
    }
    for (int i = 0; i < 77; ++i) {
        if (!used[i]) {
            empty_cell.push(i);
        }
    }
}

int MctsAgent::Node::Move(const int& idx_move, const int& idx_agent) {
    ASSERT_RANGE(idx_move, 0, n_children);
    if (node_type == NodeType::AGENT_NODE) {
        int cur = idx_move;
        int res = -1;
        for (int i = 0; i < 4; ++i) {
            if (state.geese[i].size() == 0) {
                continue;
            }
            if (i == idx_agent) {
                int idx = cur % 3;
                for (int j = 0; j < 4; ++j) {
                    if ((state.last_actions[idx_agent] ^ 2) == j) {
                        continue;
                    }
                    if(idx == 0) {
                        res = j;
                        break;
                    }
                    else {
                        idx--;
                    }
                }
            }
            cur /= 3;
        }
        ASSERT_RANGE(res, 0, 4);
        return res;
    }
    else {
        ASSERT_RANGE(idx_agent, 0, 1);
        if(!do_empty_cell()) {
            InitCell();
        }
        return empty_cell[idx_move];
    }
}

MctsAgent::Node& MctsAgent::Node::KthChildren(Stack<Node, BIG>& node_buffer, Stack<Node*, BIG>& children_buffer, const int& k) {
    ASSERT_RANGE(k, 0, n_children);
    Node* child = children_buffer[children_offset + k];
    if (child == nullptr) {
        // 領域を確保
        static std::array<int, 4> agent_action;
        for (int i = 0; i < 4; ++i) {
            if (node_type == NodeType::AGENT_NODE) {
                if (state.geese[i].size() == 0) {
                    agent_action[i] = 0;
                }
            }
            else {

            }
        }
        // node_buffer.emplace(state.NextState(k));
    }
    return *child;
}

// MctsAgent::Node::Node(hungry_geese::Stage aStage, int aIndex) : w(), n(), child_nodes() {
//     state = State(aStage, aIndex);
// }

// MctsAgent::Node::Node(State aState) : w(), n(), child_nodes() {
//     state = aState;
// }

// bool MctsAgent::push(Node aNode) {
//     // ASSERT(mNodeIndex < NodeSize, "mNodeIndex is out of range.");
//     if (mNodeIndex >= NodeSize) {
//         return false;
//     }
//     mNodes[mNodeIndex] = aNode;
//     ++mNodeIndex;
//     return true;
// }

//------------------------------------------------------------------------------
// 実行
AgentResult MctsAgent::run(const Stage& aStage, int aIndex) {
    // 0ターン目は評価値最大の行動をする
    if (aStage.mTurn == 0) {
        return solve1(aStage, aIndex);
    }
    // 探索ここから
    {
        auto state = State(aStage, aIndex);
        auto node = Node(state,children_buffer);
        // 初期化
        // mNodeIndex = 0;
        // auto root_node = Node(aStage, aIndex);
        // push(root_node);
        // expand(0);
        // シミュレーション
        // 結果を返す
        AgentResult result;

        // return result;
    }
    return solve1(aStage, aIndex);
}

//------------------------------------------------------------------------------
// 評価値最大の行動を返す
AgentResult MctsAgent::solve1(const Stage& aStage, int aIndex) {
    AgentResult result; 
    std::array<Stack<Point, 77>, 4> geese;
    std::array<Point, 2> foods;
    for (int i = 0; i < 4; ++i) {
        if (!aStage.geese()[i].isSurvive()) {
            continue;
        }
        geese[i] = aStage.geese()[i].items();
    }
    for (int i = 0; i < 2; ++i) {
        foods[i] = aStage.foods()[i].pos();
    }
    std::swap(geese[0], geese[aIndex]);
    auto res = mEvaluator.evaluate(geese, foods);
    result.mValue = res.value;
    for (int i = 0; i < 4; ++i) {
        result.mPolicy[i] = res.policy[i];
    }
    { // 前ターンの反対の行動を取らない
        auto act = aStage.mLastActions;
        if (act[aIndex] == Action::NORTH) {
            result.mPolicy[2] = -100;
        }
        else if (act[aIndex] == Action::EAST) {
            result.mPolicy[3] = -100;
        }
        else if (act[aIndex] == Action::SOUTH) {
            result.mPolicy[0] = -100;
        }
        else if (act[aIndex] == Action::WEST) {
            result.mPolicy[1] = -100;
        }
    }
    // 評価値の一番高い手を選ぶ
    int opt_action = 0;
    for (int i = 0; i < 4; ++i) {
        if (result.mPolicy[opt_action] < result.mPolicy[i]) {
            opt_action = i;
        }
    }
    result.mAction = opt_action;
    return result;
}

//------------------------------------------------------------------------------
// 方向を指定して移動先を返す関数
Point MctsAgent::Translate(Point aPos, int Direction) {
    int nx = aPos.x;
    int ny = aPos.y;
    nx += dx[Direction];
    if (nx < 0) {
        nx += Parameter::rows; 
    }
    if (nx == Parameter::rows) {
        nx = 0;
    }
    ny += dy[Direction];
    if (ny < 0) {
        ny += Parameter::columns; 
    }
    if (ny == Parameter::columns) {
        ny = 0;
    }
    return Point(nx,ny);
}

}