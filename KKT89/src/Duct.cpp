#pragma once
#include "Duct.hpp"

namespace hungry_geese {

//------------------------------------------------------------------------------
// コンストラクタ
Duct::Duct() : node_buffer(), children_buffer(), move_buffer(), model() {}

//------------------------------------------------------------------------------
// Point
Duct::Cpoint::Cpoint() : mC() {}

Duct::Cpoint::Cpoint(int aX, int aY) {
    mC = aX * hungry_geese::Parameter::columns + aY;
}

Duct::Cpoint::Cpoint(int aId) {
    mC = aId;
}

int Duct::Cpoint::X() const {
    return (int)mC / Parameter::columns;
}

int Duct::Cpoint::Y() const {
    return (int)mC % Parameter::columns;
}

int Duct::Cpoint::Id() const {
    return (int)mC;
}

Duct::Cpoint& Duct::Cpoint::operator= (const Cpoint &aPos) {
    mC = aPos.Id();
    return *this;
}

bool Duct::Cpoint::operator== (const Cpoint &aPos) const {
    return (mC == aPos.Id());
}

//------------------------------------------------------------------------------
// State
Duct::State::State() : geese(), boundary(), foods(), current_step(), last_actions() {}

Duct::State::State(hungry_geese::Stage aStage, int aIndex) : geese(), boundary(), foods(), current_step(), last_actions() {
    // Goose
    std::swap(aStage.mGeese[0],aStage.mGeese[aIndex]);
    int index = 0;
    for (int i = 0; i < 4; ++i) {
        boundary[i] = index;
        if(!aStage.mGeese[i].isSurvive()) {
            continue;
        }
        auto goose = aStage.geese()[i].items();
        for (int j = 0; j < goose.size(); ++j) {
            geese[index] = Duct::Cpoint(goose[j].id);
            index++;
        }
    }
    boundary[4] = index;
    // 食べ物
    for (int i = 0; i < 2; ++i) {
         foods[i] = Duct::Cpoint(aStage.foods()[i].pos().id);
    }
    // ターン数
    current_step = aStage.mTurn;
    // 最後の行動
    auto LastActions = aStage.mLastActions;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (LastActions[i] == Idx_to_Actions[j]) {
                last_actions += (1 << (i+i)) * j;
            }
        }
    }
}

unsigned char Duct::State::goose_size(unsigned char idx_agent) {
    return boundary[idx_agent + 1] - boundary[idx_agent];
}

Duct::State Duct::State::NextState(NodeType node_type, const unsigned char agent_action, const unsigned char food_sub) const {
    State nextstate;
    nextstate.geese = geese;
    nextstate.foods = foods;
    nextstate.current_step = current_step + 1;
    nextstate.last_actions = last_actions;
    if (node_type == NodeType::AGENT_NODE) {
        Simulate(nextstate, agent_action);
        nextstate.last_actions = agent_action;
    }
    else {
        for (int i = 0; i < 2; ++i) {
            if (foods[i].Id() == -1) {
                nextstate.foods[i] = Duct::Cpoint(agent_action);
                if ((i == 0) and (foods[i+1].Id() == -1)) {
                    nextstate.foods[i+1] = Duct::Cpoint(food_sub);
                    break;
                }
            }
        }
    }
    return nextstate;
}

void Duct::State::Simulate(State &state, unsigned char agent_action) {
    static std::array<Cpoint, 77> n_goose;
    static std::array<signed char, 5> n_boundary;
    unsigned char index = 0;
    for (unsigned char i = 0; i < 4; ++i) {
        n_boundary[i] = index;
        if (state.goose_size(i) == 0) {
            agent_action /= 4;
            continue;
        }
        auto head = Duct::Translate(state.geese[state.boundary[i]], agent_action%4);
        agent_action /= 4;
        bool eatFood = false;
        for (int j = 0; j < 2; ++j) {
            if (head == state.foods[j]) {
                eatFood = true;
                state.foods[j] = Duct::Cpoint(-1);
            }
        }
        for (int j = state.boundary[i]; j < state.boundary[i + 1]; ++j) {
            if (j + 1 == state.boundary[i + 1] and !eatFood) {
                continue;
            }
            if (head == state.geese[j]) {
                index = n_boundary[i];
                break;
            }
            n_goose[index] = state.geese[j];
        }
        if ((state.current_step + 1) % hungry_geese::Parameter::hunger_rate == 0) {
            if (n_boundary[i] !=index) {
                index--;
            }
        }
    }
    n_boundary[4] = index;
    static std::array<unsigned char, 77> simulate_goose_positions;
    for (int i = 0; i < 77; ++i) {
        simulate_goose_positions[i] = 0;
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = n_boundary[i]; j < n_boundary[i+1] ; ++j) {
            simulate_goose_positions[n_goose[j].Id()]++;
        }
    }
    index = 0;
    for (int i = 0; i < 4; ++i) {
        state.boundary[i] = index;
        if(n_boundary[i] < n_boundary[i + 1]) {
            auto head = n_goose[n_boundary[i]];
            if (simulate_goose_positions[head.Id()] == 1) {
                for (int j = n_boundary[i]; j < n_boundary[i+1] ; ++j) {
                    state.geese[index] = n_goose[j];
                    index++;
                }
            }
        }
    }
    state.boundary[4] = index;
}

//------------------------------------------------------------------------------
// Node
std::array<std::array<float, 4>, 4> Duct::Node::GetPolicy() const {
    return policy;
}

std::array<std::array<float, 4>, 4> Duct::Node::GetWorth() const {
    return worth;
}

Duct::Node::Node() : state(), policy(), worth(), n(0), n_children(), children_offset(), node_type() {}

Duct::Node::Node(const State& aState, Stack<Node*, BIG>& children_buffer) : state(aState), policy(), worth(), n(0), children_offset(), node_type() {
    policy[0][0] = -100.0;

    if (aState.foods[0].Id() == -1 or aState.foods[1].Id() == -1) {
        node_type = NodeType::FOOD_NODE;
    }

    // 子ノードの数を数える処理
    if (node_type == NodeType::AGENT_NODE) {
        n_children = 1;
        for (unsigned char i = 0; i < 4; ++i) {
            if (aState.boundary[i] != aState.boundary[i + 1]) {
                n_children *= 3;
            }
        }
    }
    else {
        n_children = 77;
        for (int i = 0; i < 2; ++i) {
            if (aState.foods[i].Id() != -1) {
                n_children--;
            }
        }
        n_children -= aState.boundary[4];
        if (aState.foods[0].Id() == -1 and aState.foods[1].Id() == -1) {
            n_children = n_children * (n_children - 1) / 2;
        }
    }

    children_offset = children_buffer.size();
    children_buffer.resize(children_offset + n_children);
}

bool Duct::Node::Expanded() const {
    return (policy[0][0] != -100.0);
}

int Duct::Node::ChooseMove() {
    // 未実装
    return 0;
}

// int Duct::Node::Move(const int& idx_move, const int& idx_agent) {
//     ASSERT_RANGE(idx_move, 0, n_children);
//     if (node_type == NodeType::AGENT_NODE) {
//         int cur = idx_move;
//         int res = -1;
//         for (int i = 0; i < 4; ++i) {
//             if (state.goose_size((unsigned char) i) == 0) {
//                 continue;
//             }
//             // if (i == idx_agent) {
//             //     int idx = cur % 3;
//             //     for (int j = 0; j < 4; ++j) {
//             //         if ((state.last_actions[idx_agent] ^ 2) == j) {
//             //             continue;
//             //         }
//             //         if(idx == 0) {
//             //             res = j;
//             //             break;
//             //         }
//             //         else {
//             //             idx--;
//             //         }
//             //     }
//             // }
//             // cur /= 3;
//         }
//         ASSERT_RANGE(res, 0, 4);
//         return res;
//     }
//     // else {
//     //     ASSERT_RANGE(idx_agent, 0, 1);
//     //     if(!DoInitCell()) {
//     //         InitCell();
//     //     }
//     //     return empty_cell[idx_move];
//     // }
// }

Duct::Node& Duct::Node::KthChildren(Stack<Node, BIG>& node_buffer, Stack<Node*, BIG>& children_buffer, const int& k) {
    ASSERT_RANGE(k, 0, n_children);
    Node* child = children_buffer[children_offset + k];
    if (child == nullptr) {
        // 領域を確保
        unsigned char agent_action = 0;
        State nextstate;
        if (node_type == NodeType::AGENT_NODE) {
            unsigned char idx_move = k;
            for (int i = 0; i < 4; ++i) {
                if (state.goose_size((unsigned char) i) == 0) {
                    continue;
                }
                else {
                    unsigned char ith_idx_move = idx_move % 3;
                    unsigned char ith_idx_lastmove = 0;
                    if (state.last_actions & (1 << (i + i))) ith_idx_lastmove++;
                    if (state.last_actions & (1 << (i + i + 1))) ith_idx_lastmove+=2;
                    idx_move /= 3;
                    for (int j = 0; j < 4; ++j) {
                        if ((ith_idx_lastmove ^ 2) == j) {
                            continue;
                        }
                        if (ith_idx_move == 0) {
                            agent_action += (j << (i + i));
                            break;
                        }
                        ith_idx_move--;
                    }
                }
            }
            nextstate = state.NextState(node_type, agent_action, 0);
        }
        else {
            nextstate = state;
            static std::array<bool, 77> used;
            for (int i = 0; i < 77; ++i) {
                used[i] = false;
            }
            int empty_cell = 77 - state.boundary[4];
            for (int i = 0; i < state.boundary[4]; ++i) {
                used[state.geese[i].Id()] = true;
            }
            for (int i = 0; i < 2; ++i) {
                if (state.foods[i].Id() != -1) {
                    empty_cell--;
                    used[state.foods[i].Id()] = true;
                }
            }
            // 空きマスがN個あって、2つ空きマスを選ぶのはN*(N-1)/2
            // k = [0,N*(N-1)/2) → 空きマス二つを選ぶ
            int idx_move = k;
            if (empty_cell < n_children) { // 2個選ぶ場合
                for (int i = 0; i < 77; ++i) {
                    if (used[i]) {
                        continue;
                    }
                    if (idx_move < empty_cell) {
                        nextstate.foods[0] = Cpoint(i);
                        for (int j = i + 1; j < 77; ++j) {
                            if (used[j]) {
                                continue;
                            }
                            if (idx_move == 0) {
                                nextstate.foods[1] = Cpoint(j);
                                break;
                            }
                            else {
                                idx_move--;
                            }
                        }
                        break;
                    }
                    else {
                        empty_cell--;
                        idx_move -= empty_cell;
                    }
                }
            }
            else { // 1個選ぶ場合
                for (int i = 0; i < 77; ++i) {
                    if (used[i]) {
                        continue;
                    }
                    if (idx_move == 0) {
                        if (state.foods[0].Id() == -1) {
                            nextstate.foods[0] = Cpoint(i);
                        }
                        else {
                            nextstate.foods[1] = Cpoint(i);
                        }
                        break;
                    }
                    idx_move--;
                }
            }
        }
        auto nextnode = Node(nextstate, children_buffer);
        node_buffer.emplace(nextnode);
        child = children_buffer[children_offset + k] = &node_buffer.back();
    }
    return *child;
}

//------------------------------------------------------------------------------
// コンストラクタ
Duct::Duct(const Node& arg_state) {
    node_buffer[0] = arg_state;
    for (auto&& c : children_buffer) c = nullptr;
}
// 初期化
void Duct::InitDuct(const Node& arg_state) {
    node_buffer[0] = arg_state;
    for (auto&& c : children_buffer) c = nullptr;
}

//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// 方向を指定して移動先を返す関数
Duct::Cpoint Duct::Translate(Cpoint aPos, int Direction) {
    int nx = aPos.X();
    int ny = aPos.Y();
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
    return Cpoint(nx,ny);
}

}

