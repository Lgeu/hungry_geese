#pragma once
#include "Duct.hpp"

namespace hungry_geese {

//------------------------------------------------------------------------------
// コンストラクタ
Duct::Duct() : node_buffer(), children_buffer(), model() {}

//------------------------------------------------------------------------------
// Ductのログをデバック出力
void Duct::Setprintlog(bool f) {
    printlog = f;
}

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
Duct::State::State() : geese(), boundary(), foods(), current_step(), last_actions(), ranking() {}

Duct::State::State(hungry_geese::Stage aStage, int aIndex) : geese(), boundary(), foods(), current_step(), last_actions(), ranking() {
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
    std::swap(LastActions[0], LastActions[aIndex]);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (LastActions[i] == Idx_to_Actions[j]) {
                last_actions += (1 << (i+i)) * j;
            }
        }
    }
    // 順位情報
    for (int i = 0; i < 4; ++i) {
        if (boundary[i + 1] - boundary[i] == 0) {
            ranking[i] = 4;
        }
        else {
            // 未確定は0
            ranking[i] = 0;
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
    nextstate.ranking = ranking;
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
    static std::array<signed char, 4> pre_gooselength;
    unsigned char index = 0;
    for (unsigned char i = 0; i < 4; ++i) {
        n_boundary[i] = index;
        pre_gooselength[i] = state.boundary[i + 1] - state.boundary[i];
        if (pre_gooselength[i] == 0) {
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
    for (int i = 0; i < 4; ++i) {
        // この行動によって脱落したAgentの順位付けをする
        if (pre_gooselength[i] != 0 and state.boundary[i + 1] - state.boundary[i] == 0) {
            unsigned char rank = 1;
            for (int j = 0; j < 4; ++j) {
                if (i == j) {
                    continue;
                }
                else if (state.boundary[j + 1] - state.boundary[j] != 0) {
                    rank++;
                }
                else if (pre_gooselength[j] < pre_gooselength[i]) {
                    rank++;
                }
            }
            state.ranking[i] = rank;
        }
    }
}

bool Duct::State::Finished() const {
    int surviver = 0;
    for (int i = 0; i < 4; ++i) {
        if (boundary[i + 1] - boundary[i] > 0) {
            surviver++;
        }
    }
    if(surviver <= 1) {
        return true;
    }
    else {
        return false;
    }
}

void Duct::State::Debug() const {
    // lastmove
    int act = last_actions;
    std::cerr << "last_actions";
    for (int i = 0; i < 4; ++i) {
        std::cerr << " " << act%4;
        act /= 4;
    }
    std::cerr << std::endl;
}

//------------------------------------------------------------------------------
// Node
const std::array<std::array<float, 4>, 4>& Duct::Node::GetPolicy() const {
    return policy;
}

const std::array<std::array<float, 4>, 4>& Duct::Node::GetWorth() const {
    return worth;
}

Duct::Node::Node() : state(), policy(), value(), worth(), n(), n_children(), children_offset(), node_type() {}

Duct::Node::Node(const State& aState, Stack<Node*, children_buffer_size>& children_buffer) : state(aState), policy(), value(), worth(), n(), children_offset(), node_type() {
    policy[0][0] = -100.0;
    value[0] = -100.0;

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
    children_buffer.resize(children_offset + n_children, nullptr);
}

bool Duct::Node::Expanded() const {
    return (policy[0][0] != -100.0);
}

float Duct::Node::Argvalue(const int& idx_agent, const int& idx_move, const int& t_sum) {
    constexpr float c_puct = 1.0;
    float n_sum = 1e-1 + n[idx_agent][0] + n[idx_agent][1] + n[idx_agent][2] + n[idx_agent][3];
    return GetWorth()[idx_agent][idx_move] / (float)(1e-1 + n[idx_agent][idx_move]) + c_puct * GetPolicy()[idx_agent][idx_move] * std::sqrt(t_sum) / (float)(1 + n[idx_agent][idx_move]);
}

int Duct::Node::ChooseMove(const int& t_sum) {
    int k = 0;
    if (node_type == NodeType::AGENT_NODE) {
        unsigned char base = 1;
        for (int i = 0; i < 4; ++i) {
            if (state.goose_size(i) == 0) {
                continue;
            }
            unsigned char ith_idx_lastmove = 0;
            if (state.last_actions & (1 << (i + i))) ith_idx_lastmove++;
            if (state.last_actions & (1 << (i + i + 1))) ith_idx_lastmove+=2;
            float maxvalue = -100.0;
            unsigned char opt_action = 0;
            for (int j = 0; j < 4; ++j) {
                if ((ith_idx_lastmove ^ 2) == j) {
                    continue;
                }
                auto res = Argvalue(i, j, t_sum);
                if (res > maxvalue) {
                    maxvalue = res;
                    opt_action = j;
                }
            }
            if (opt_action > (ith_idx_lastmove ^2)) {
                opt_action--;
            }
            k += base * opt_action;
            base *= 3;
        }
    }
    else {
        static std::mt19937 engine(std::chrono::steady_clock::now().time_since_epoch().count());
        return engine() % n_children;
    }
    return k;
}

Duct::Node& Duct::Node::KthChildren(Stack<Node, node_buffer_size>& node_buffer, Stack<Node*, children_buffer_size>& children_buffer, const int& k) {
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
        node_buffer.push(nextnode);
        child = children_buffer[children_offset + k] = &node_buffer.back();
    }
    return *child;
}

void Duct::Node::Debug() const {
    // State
}

//------------------------------------------------------------------------------
// コンストラクタ
Duct::Duct(const Node& arg_state) {
    node_buffer[0] = arg_state;
    for (auto&& c : children_buffer) c = nullptr;
    t_sum = 0;
}
// 初期化
void Duct::InitDuct(const Node& arg_state) {
    node_buffer.clear();
    node_buffer.push(arg_state);
    children_buffer.clear();
    t_sum = 0;
}
void Duct::InitDuct(hungry_geese::Stage aStage, int aIndex) {
    children_buffer.clear();
    t_sum = 0;
    auto state = Duct::State(aStage, aIndex);
    auto node = Duct::Node(state, children_buffer);
    node_buffer.clear();
    node_buffer.push(node);
    if (printlog) {
        // ターン情報
        std::cout << "Turn : " << aStage.mTurn << " " << "Agent : " << aIndex << std::endl;
        state.Debug();
    }
}

//------------------------------------------------------------------------------
void Duct::Search(const float timelimit) {
    double timebegin = nagiss_library::time();
    while (nagiss_library::time() - timebegin < timelimit) {
        Iterate();
        t_sum++;
    }
}

Duct::Node& Duct::RootNode() {
    return node_buffer[0];
}

void Duct::Iterate() {
    // 根から葉に移動
    Node* v = &RootNode();
    Stack<int, 100> path;
    // 展開されてない、エージェントのノードに到達したら抜ける
    while (v->Expanded() or v->node_type == NodeType::FOOD_NODE) {
        int move_idx = v->ChooseMove(t_sum);
        path.push(move_idx);
        v = &v->KthChildren(node_buffer, children_buffer, move_idx);
        if (v->state.Finished()) { // 終局状態
            break;
        }
    }

    // 葉ノードの処理
    std::array<float, 4> value;
    if (v->state.Finished()) {
        // 決着がついた場合、順位に応じて value を設定
        for (int i = 0; i < 3; ++i) {
            for (int j = i + 1; j < 4; ++j) {
                if (v->state.ranking[i] < v->state.ranking[j]) {
                    value[i]++;
                }
                else if (v->state.ranking[i] == v->state.ranking[j]) {
                    value[i] += 0.5f;
                    value[j] += 0.5f;
                }
                else {
                    value[j]++;
                }
            }
        }
    }
    else {
        Node* leaf = v;
        std::array<Stack<int, 77>, 4> geese;
        std::array<int, 2> foods;
        for (int i = 0; i < 4; ++i) {
            for (int j = v->state.boundary[i]; j < v->state.boundary[i + 1]; ++j) {
                geese[i].push(v->state.geese[j].Id());
            }
        }
        for (int i = 0; i < 2; ++i) {
            foods[i] = v->state.foods[i].Id();
        }
        for (int i = 0; i < 4; ++i) {
            std::swap(geese[0], geese[i]);
            auto res = model.evaluate(geese, foods);
            for (int j = 0; j < 4; ++j) {
                v->policy[i][j] = res.policy[j];
            }
            v->value[i] = res.value;
            std::swap(geese[0], geese[i]);
        }
    }

    // 葉までの評価結果を経路のノードに反映
    v = &RootNode();
    for (const auto& move_idx : path) {
        int k = move_idx;
        for (int idx_agent = 0; idx_agent < 4; ++idx_agent) {
            if (v->state.goose_size(idx_agent) == 0) {
                continue;
            }
            unsigned char opt_action = k % 3;
            k /= 3; 
            unsigned char ith_idx_lastmove = 0;
            if (v->state.last_actions & (1 << (idx_agent + idx_agent))) ith_idx_lastmove++;
            if (v->state.last_actions & (1 << (idx_agent + idx_agent + 1))) ith_idx_lastmove+=2;
            if (opt_action >= (ith_idx_lastmove ^2)) {
                opt_action++;
            }
            v->worth[idx_agent][opt_action] += value[idx_agent];
            v->n[idx_agent][opt_action]++;
        }
        v = &v->KthChildren(node_buffer, children_buffer, move_idx);
    }
}

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

