#pragma once
#include "Duct.hpp"

namespace hungry_geese {

//------------------------------------------------------------------------------
// コンストラクタ
Duct::Duct() : node_buffer(), children_buffer(), nnue() {}

//------------------------------------------------------------------------------
// Ductのログをデバック出力
void Duct::Setprintlog(bool f) {
    printlog = f;
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
            geese[index] = goose[j];
            index++;
        }
    }
    boundary[4] = index;
    // 食べ物
    for (int i = 0; i < 2; ++i) {
         foods[i] = aStage.foods()[i].pos();
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
    State nextstate(*this);
    if (node_type == NodeType::AGENT_NODE) {
        Simulate(nextstate, agent_action);
    }
    else {
        for (int i = 0; i < 2; ++i) {
            if (foods[i].Id() == -1) {
                nextstate.foods[i] = Cpoint(agent_action);
                if ((i == 0) and (foods[i+1].Id() == -1)) {
                    nextstate.foods[i+1] = Cpoint(food_sub);
                    break;
                }
            }
        }
    }
    return nextstate;
}

void Duct::State::Simulate(State &state, const unsigned char& agent_actions) {
    // state のメンバ変数である geese, boundary, foods, current_step, last_actions, ranking を更新する。
    // foods は食べても補充はしない。
    // agent_actionsは 4 進数


    // かっつ すまん…


    // current_step の更新 (これだけ先)
    state.current_step++;

    // 行動選択時点での goose の長さ
    auto last_lengths = std::array<int, 4>();

    // geese のコピー
    static auto geese = std::array<nagiss_library::Stack<Cpoint, 77>, 4>();
    for (auto idx_geese = 0; idx_geese < 4; idx_geese++) {
        geese[idx_geese].clear();
        for (auto j = state.boundary[idx_geese]; j < state.boundary[idx_geese + 1]; j++) {
            geese[idx_geese].push(state.geese[j]);
        }
        last_lengths[idx_geese] = state.boundary[idx_geese + 1] - state.boundary[idx_geese];
        if (last_lengths[idx_geese] == 0) {
            ASSERT(state.ranking[idx_geese] > 0, "長さが 0 ならもう順位が確定しているはずだよ");
        }
        else {
            ASSERT(state.ranking[idx_geese] <= 0, "長さが 0 じゃないのに順位が確定してるよ");
        }
    }
    // foods のコピー
    auto foods = nagiss_library::Stack<Cpoint, 2>();
    for (const auto& p : state.foods) {
        ASSERT(p.Id() != -1, "なんか食べ物が -1 だよ");
        foods.push(p);
    }

    // 各エージェントの処理
    for (int idx_geese = 0; idx_geese < 4; idx_geese++) {
        if (state.ranking[idx_geese] != 0) continue;
        const auto action = (agent_actions >> 2 * idx_geese) & 3;
        const auto last_action = (state.last_actions >> 2 * idx_geese) & 3;
        ASSERT(action != (last_action ^ 2), "逆向きの行動は取らないはずだよ");
        
        auto& goose = geese[idx_geese];
        ASSERT(goose.size() > 0, "脱落してるのに ranking が 0 だよ");
        const auto head = Duct::Translate(goose[0], action);

        // 食べ物
        if (foods.contains(head)) {
            foods.remove(head);
        }
        else {
            goose.pop();
        }

        // 自己衝突
        if (goose.contains(head)) {
            goose.clear();
        }

        // 頭をつける
        goose.insert(0, head);

        // おなかすいた
        if (state.current_step % hungry_geese::Parameter::hunger_rate == 0) {
            if (goose.size() > 0) {
                goose.pop();
            }
        }
    }
    
    static auto goose_positions = std::array<int, 77>();
    std::fill(goose_positions.begin(), goose_positions.end(), 0);
    for (const auto& goose : geese)
        for (const auto& position : goose)
            goose_positions[position.Id()]++;

    // 相互衝突
    for (int idx_geese = 0; idx_geese < 4; idx_geese++) {
        auto& goose = geese[idx_geese];
        if (goose.size() > 0) {
            const auto& head = goose[0];
            if (goose_positions[head.Id()] > 1) {
                goose.clear();
            }
        }
    }

    // ranking の更新
    for (int idx_geese = 0; idx_geese < 4; idx_geese++) {
        if (last_lengths[idx_geese] > 0 && geese[idx_geese].size() == 0) {  // このステップに脱落した
            int rank = 1;
            for (int opponent = 0; opponent < 4; opponent++) {
                if (idx_geese == opponent) {
                    continue;
                }
                // 相手がまだ生きてる or 同時に脱落したけど自分より長い なら負け
                if (geese[opponent].size() > 0 || last_lengths[opponent] > last_lengths[idx_geese]) {
                    rank++;
                }
            }
            state.ranking[idx_geese] = rank;
        }
    }

    // geese, boundary の更新
    ASSERT(state.boundary[0] == 0, "オイオイオイ 0 じゃないわこいつ");
    int idx_geese_series = 0;
    for (int idx_geese = 0; idx_geese < 4; idx_geese++) {
        const auto& goose = geese[idx_geese];
        ASSERT(goose.size() >= 0, "んん");

        for (const auto& p : goose) {
            state.geese[idx_geese_series++] = p;
        }
        state.boundary[idx_geese + 1] = idx_geese_series;
    }

    // foods の更新 (補充しない)
    fill(state.foods.begin(), state.foods.end(), Cpoint(-1));
    for (int idx_foods = 0; idx_foods < foods.size(); idx_foods++) {
        state.foods[idx_foods] = foods[idx_foods];
    }

    // last_actions の更新
    state.last_actions = agent_actions;
}

bool Duct::State::Finished() const {
    int surviver = 0;
    for (int i = 0; i < 4; ++i) {
        if (boundary[i + 1] - boundary[i] > 0) {
            surviver++;
        }
    }
    if(surviver <= 1 or current_step >= 199) {
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

Duct::Node::Node(const State& aState, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer) : state(aState), policy(), value(), worth(), n(), children_offset(), node_type() {
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
    children_buffer.resize(children_offset + n_children, nullptr);
}

bool Duct::Node::Expanded() const {
    return (policy[0][0] != -100.0);
}

float Duct::Node::Argvalue(const int& idx_agent, const int& idx_move, const int& t_sum) {
    constexpr float c_puct = 1.0f;
    return (3.0f + GetWorth()[idx_agent][idx_move]) / (1.0f + (float)n[idx_agent][idx_move]) 
        + c_puct * GetPolicy()[idx_agent][idx_move] * std::sqrtf(t_sum) / (float)(1 + n[idx_agent][idx_move]);
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

Duct::Node& Duct::Node::KthChildren(nagiss_library::Stack<Node, node_buffer_size>& node_buffer, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer, const int& k) {
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
}

//------------------------------------------------------------------------------
AgentResult Duct::Search(const float timelimit) {
    AgentResult result;
    if (node_buffer[0].state.current_step == 0) {
        std::array<nagiss_library::Stack<int, 77>, 4> geese;
        std::array<int, 2> foods;
        std::array<int, 4> rank;
        auto Convert_CpointArray_to_IntArray=[](const State &state, std::array<nagiss_library::Stack<int, 77>, 4> &geese, std::array<int, 2>& foods, std::array<int, 4> &rank)->void{
            for (int i = 0; i < 4; ++i) {
                for (int j = state.boundary[i]; j < state.boundary[i + 1]; ++j) {
                    geese[i].push(state.geese[j].Id());
                }
            }
            for (int i = 0; i < 2; ++i) {
                foods[i] = state.foods[i].Id();
            }
            for (int i = 0; i < 4; ++i) {
                rank[i] = state.ranking[i];
            }
        };
        auto rootnode = RootNode();
        Convert_CpointArray_to_IntArray(rootnode.state, geese, foods, rank);
        auto res = nnue.evaluate(geese, foods, rootnode.state.current_step, rank);
        result.mValue = res[0].value;
        for (int i = 0; i < 4; ++i) {
            result.mPolicy[i] = res[0].policy[i];
        }
    }
    else {
        double timebegin = nagiss_library::time();
        while (nagiss_library::time() - timebegin < timelimit) {
            Iterate();
            t_sum++;
        }
        auto rootnode = RootNode();
        result.mValue = rootnode.value[0];
        for (int i = 0; i < 4; ++i) {
            result.mPolicy[i] = (float)rootnode.n[0][i] / (float)(rootnode.n[0][0] + rootnode.n[0][1] + rootnode.n[0][2] + rootnode.n[0][3]);
        }
    }
    unsigned char opt_action = 0;
    for (int i = 0; i < 4; ++i) {
        if (result.mPolicy[opt_action] < result.mPolicy[i]) {
            opt_action = i;
        }
    }
    result.mAction = opt_action;
    // std::cerr << (int)node_buffer[0].state.current_step << " " << t_sum << " " << node_buffer.size() << " " << children_buffer.size() << std::endl;
    return result;
}

Duct::Node& Duct::RootNode() {
    return node_buffer[0];
}

void Duct::Iterate() {
    // 根から葉に移動
    Node* v = &RootNode();
    nagiss_library::Stack<int, 200> path;
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
    std::array<float, 4> value{};
    if (v->state.Finished()) {
        for (int i = 0; i < 4; ++i) {
            // 生き残ってる
            if (v->state.boundary[i + 1] > v->state.boundary[i]) {
                int rank = 1;
                for (int j = 0; j < 4; ++j) {
                    if (i == j) {
                        continue;
                    }
                    if (v->state.boundary[j + 1] - v->state.boundary[j] > v->state.boundary[i + 1] - v->state.boundary[i]) {
                        rank++;
                    }
                }
                v->state.ranking[i] = rank;
            }
        }
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
        Node* const leaf = v;
        std::array<nagiss_library::Stack<int, 77>, 4> geese;
        std::array<int, 2> foods;
        std::array<int, 4> rank;
        auto Convert_CpointArray_to_IntArray=[](const State &state, std::array<nagiss_library::Stack<int, 77>, 4> &geese, std::array<int, 2>& foods, std::array<int, 4> &rank)->void{
            for (int i = 0; i < 4; ++i) {
                for (int j = state.boundary[i]; j < state.boundary[i + 1]; ++j) {
                    geese[i].push(state.geese[j].Id());
                }
            }
            for (int i = 0; i < 2; ++i) {
                foods[i] = state.foods[i].Id();
            }
            for (int i = 0; i < 4; ++i) {
                rank[i] = state.ranking[i];
            }
        };
        Convert_CpointArray_to_IntArray(v->state, geese, foods, rank);
        auto res = nnue.evaluate(geese, foods, v->state.current_step, rank);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v->policy[i][j] = res[i].policy[j];
            }
            value[i] = res[i].value;
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

    /*
    if (RootNode().state.current_step >= 80 && t_sum >= 1000) {
        int zero_counts = 0;
        int sum_n = 0;
        for (int i = 0; i < 4; i++) {
            zero_counts += RootNode().n[0][i] == 0;
            sum_n += RootNode().n[0][i];
        }
        
        if (zero_counts >= 3) {
            for (int mv = 0; mv < 4; mv++) {
                std::cerr << RootNode().Argvalue(0, mv, t_sum) << " ";
            }
            std::cerr << "???" << std::endl;
        }
        
        bool f = false;
        for (int mv = 0; mv < 4; mv++) {
            // 探索後の policy がモデルの policy と比べて極端に大きい
            auto policy_after_search = (float)RootNode().n[0][mv] / (float)sum_n;
            if (policy_after_search >= 0.4f && policy_after_search / RootNode().policy[0][mv] >= 10.0f) {
                f = true;
            }
        }
        if (f) {
            std::cerr << "!?!?" << std::endl;
        }
    }
    */
}

//------------------------------------------------------------------------------
// 方向を指定して移動先を返す関数
Cpoint Duct::Translate(Cpoint aPos, int Direction) {
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

