#pragma once
#include "Duct2.hpp"

namespace hungry_geese {

//------------------------------------------------------------------------------
// コンストラクタ
Duct2::Duct2() : node_buffer(), children_buffer(), nnue() {}

//------------------------------------------------------------------------------
// State
Duct2::State::State() : geese(), boundary(), foods(), current_step(), last_actions(), ranking() {}

Duct2::State::State(hungry_geese::Stage aStage, int aIndex) : geese(), boundary(), foods(), current_step(), last_actions(), ranking() {
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

signed char Duct2::State::goose_size(signed char idx_agent) {
    return boundary[idx_agent + 1] - boundary[idx_agent];
}

Duct2::State Duct2::State::NextState(NodeType node_type, const unsigned char agent_action, const unsigned char food_sub) const {
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

void Duct2::State::Simulate(State &state, const unsigned char& agent_actions) {
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
        // ASSERT(p.Id() != -1, "なんか食べ物が -1 だよ");
        // 7/23修正：-1なことがあるように
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
        const auto head = Duct2::Translate(goose[0], action);

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
            continue;
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
    std::fill(state.foods.begin(), state.foods.end(), Cpoint(-1));
    for (int idx_foods = 0; idx_foods < foods.size(); idx_foods++) {
        state.foods[idx_foods] = foods[idx_foods];
    }

    // last_actions の更新
    state.last_actions = agent_actions;
}

bool Duct2::State::Finished() const {
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

void Duct2::State::Debug() const {
    // Goose
    for (int i = 0; i < 4; ++i) {
        std::cerr << boundary[i + 1] - boundary[i];
        for (int j = boundary[i]; j < boundary[i + 1]; ++j) {
            std::cerr << " " << (int)(geese[j].Id());
        }
        std::cerr << std::endl;
    }
    // food
    std::cerr << (int)(foods[0].Id()) << " " << (int)(foods[1].Id()) << std::endl;
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
const std::array<std::array<float, 4>, 4>& Duct2::Node::GetWorth() const {
    return worth;
}

Duct2::Node::Node() : state(), value(), worth(), n(), visited(), n_children(), children_offset(), node_type() {}

Duct2::Node::Node(const State& aState, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer) : state(aState), value(), worth(), n(), visited(), n_children(), children_offset(), node_type() {

    if (aState.foods[0].Id() == -1 or aState.foods[1].Id() == -1) {
        node_type = NodeType::FOOD_NODE;
    }

    // 子ノードの数を数える処理

    // 7/23修正；NodeType::FOOD_NODEを先に処理
    if (node_type == NodeType::FOOD_NODE) {
        int empty_cell = 77;
        for (int i = 0; i < 2; ++i) {
            if (aState.foods[i].Id() != -1) {
                empty_cell--;
            }
        }
        empty_cell -= aState.boundary[4];

        // 7/23修正：NodeType::FOOD_NODEなのに食べ物が置けないことがあるのを直したい

        // 2個食べ物を補充したい
        if (aState.foods[0].Id() == -1 and aState.foods[1].Id() == -1) {
            // 問題なく2個置ける
            if (empty_cell >= 2) {
                n_children = empty_cell * (empty_cell - 1) / 2;
            }
            else if (empty_cell == 1) {
                // 1個だけ置ける
                // 子ノードの数は 1 (= empty_cell)
                n_children = empty_cell;
            }
            else {
                // 1個も置けない　つまりAGENT_NODE
                node_type = NodeType::AGENT_NODE;
            }
        }
        // 1個食べ物を補充したい
        else {
            // 問題なく1個置ける
            if (empty_cell >= 1) {
                // このままでOK
                n_children = empty_cell;
            }
            else {
                // 1個も置けない　つまりAGENT_NODE
                node_type = NodeType::AGENT_NODE;
            }
        }

    }
    // else if でなく if にする
    // 上の処理中で NodeType::AGENT_NODE になることがあるため
    if (node_type == NodeType::AGENT_NODE) {
        n_children = 1;
        for (unsigned char i = 0; i < 4; ++i) {
            if (aState.boundary[i] != aState.boundary[i + 1]) {
                n_children *= 3;
            }
        }
    }

    // ここを修正するとKthChildrenも見直す必要がある
    // ぱっとみた感じ修正入らない気がした
    // 次に修正すべきは食べ物いずれかが-1のAGENT_NODE時にmodelを呼び出さないようにする
    // これはDuct2::Iterateを見直す

    ASSERT(n_children > 0, "n_childrenが0だよ");

    children_offset = children_buffer.size();
    // 7/24
    // 食べ物のノード減らすためにはここを修正する必要がある
    children_buffer.resize(children_offset + SmallChildrenSize(), nullptr);
}

int Duct2::Node::SmallChildrenSize() const {
    if (node_type == NodeType::AGENT_NODE) {
        return n_children;
    }
    else {
        return std::min(10, n_children);
    }
}

float Duct2::Node::Argvalue(const int& idx_agent, const int& idx_move, const int& t_sum) {
    constexpr float c_puct = 1.0f;
    return (GetWorth()[idx_agent][idx_move]) / (1.0f + (float)n[idx_agent][idx_move]) 
        + std::pow((2.0f * std::log(t_sum)) / (1.0f + (float)n[idx_agent][idx_move]) ,0.5);
}

// 7/24修正：返り値が{k番目の子, k番目のposition_id}に
// KthChildrenに渡す順番と同一にした
// pair廃止！！！
int Duct2::Node::ChooseMove(const int& t_sum, const int& cnt) {
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
        return k;
    }
    else {
        // 7/24：ここで食べ物のNodeの選び方の枝刈りをしたいなぁ

        // とりあえず候補を3つ以下に絞る実装
        auto Pick_one_of_the_three=[&]()->int{
            return cnt % std::min(3, n_children);
        };
        
        // log(訪れた回数) を種類数とするようなイメージの実装(まだ実装考えてない)
        auto Pick_from_log=[&]()->int{
            if (cnt <= 1) {
                return 0;
            }
            if ((1 << SmallChildrenSize()) * SmallChildrenSize() <= cnt) {
                return (cnt - (1 << SmallChildrenSize()) * SmallChildrenSize()) % SmallChildrenSize();
            }
            for (int i = 1; i <= 10; ++i) {
                if ((1 << i) * i > cnt) {
                    // 種類数がi
                    if ((1 << (i-1)) * i > cnt) {
                        return i-1;
                    }
                    else {
                        return (cnt - (1 << (i-1)) * i) % i;
                    }
                }
            }
            return 0;
        };

        return Pick_one_of_the_three();
        //return Pick_from_log();
    }
}

// 7/24：修正　Nodeのk番目の子、k番目のNodeのpositions_id(次の状態が一意に定まる変数)を渡すようにする
Duct2::Node& Duct2::Node::KthChildren(nagiss_library::Stack<Node, node_buffer_size>& node_buffer, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer, const int& k, nagiss_library::PermutationGenerator generator) {
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
            int empty_food = 0;
            for (int i = 0; i < state.boundary[4]; ++i) {
                used[state.geese[i].Id()] = true;
            }
            for (int i = 0; i < 2; ++i) {
                if (state.foods[i].Id() != -1) {
                    empty_cell--;
                    used[state.foods[i].Id()] = true;
                }
                else {
                    empty_food++;
                }
            }
            // 空きマスがN個あって、2つ空きマスを選ぶのはN*(N-1)/2
            // k = [0,N*(N-1)/2) → 空きマス二つを選ぶ

            // 7/24修正
            // ここでidx_moveを計算する
            int idx_move = generator.GetKth((unsigned long long)children_offset * (unsigned long long)n_children + 9982443531000000007ull, n_children, k + 1);
            if (empty_food == 2 and 2 <= empty_cell) { // 2個選ぶ場合
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

//------------------------------------------------------------------------------
// 初期化
void Duct2::InitDuct(hungry_geese::Stage aStage, int aIndex) {
    children_buffer.clear();
    t_sum = 0;
    auto state = Duct2::State(aStage, aIndex);
    auto node = Duct2::Node(state, children_buffer);
    node_buffer.clear();
    node_buffer.push(node);
}

//------------------------------------------------------------------------------
AgentResult Duct2::Search(const float timelimit) {
    AgentResult result;
    // turn == 0 の時は考えてない
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

Duct2::Node& Duct2::RootNode() {
    return node_buffer[0];
}

void Duct2::Iterate() {
    // 根から葉に移動
    Node* v = &RootNode();
    nagiss_library::Stack<int, 200> path;
    while (true) {
        if (v->node_type == NodeType::AGENT_NODE) {
            if (v->state.Finished()) break;
            if (v->visited < 100) break; // 20回をとりあえず閾値に
        }
        // break されていなければ次のノードに行く
        int move_idx = v->ChooseMove(t_sum, v->visited);
        path.push(move_idx);

        // 7/23；Nodeに訪れた回数を記録
        // 多分ここで加算すれば良いと思ってる……
        v->visited++;

        v = &v->KthChildren(node_buffer, children_buffer, move_idx, p_generator);
    }

    // 抜けたところ(=リーフノード)でも加算
    v->visited++;

    // Playout
    auto value = Playout(v->state);

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
        v = &v->KthChildren(node_buffer, children_buffer, move_idx, p_generator);
    }
}

std::array<float ,4> Duct2::Playout(State v) const {
    static std::mt19937 rand(std::chrono::steady_clock::now().time_since_epoch().count());
    std::array<unsigned char, 4> ranking;
    for (int i = 0; i < 4; ++i) {
        ranking[i] = v.ranking[i];
    }
    while (!v.Finished()) {
        NodeType node_type = NodeType::AGENT_NODE;
        int n_children;
        if (v.foods[0].Id() == -1 or v.foods[1].Id() == -1) {
            node_type = NodeType::FOOD_NODE;
        }
        int empty_cell = 77;
        if (node_type == NodeType::FOOD_NODE) {
            for (int i = 0; i < 2; ++i) {
                if (v.foods[i].Id() != -1) {
                    empty_cell--;
                }
            }
            empty_cell -= v.boundary[4];

            // 7/23修正：NodeType::FOOD_NODEなのに食べ物が置けないことがあるのを直したい

            // 2個食べ物を補充したい
            if (v.foods[0].Id() == -1 and v.foods[1].Id() == -1) {
                // 問題なく2個置ける
                if (empty_cell >= 2) {
                    n_children = empty_cell * (empty_cell - 1) / 2;
                }
                else if (empty_cell == 1) {
                    // 1個だけ置ける
                    // 子ノードの数は 1 (= empty_cell)
                    n_children = empty_cell;
                }
                else {
                    // 1個も置けない　つまりAGENT_NODE
                    node_type = NodeType::AGENT_NODE;
                }
            }
            // 1個食べ物を補充したい
            else {
                // 問題なく1個置ける
                if (empty_cell >= 1) {
                    // このままでOK
                    n_children = empty_cell;
                }
                else {
                    // 1個も置けない　つまりAGENT_NODE
                    node_type = NodeType::AGENT_NODE;
                }
            }
        }
        if (node_type == NodeType::AGENT_NODE) {
            n_children = 1;
            for (unsigned char i = 0; i < 4; ++i) {
                if (v.boundary[i] != v.boundary[i + 1]) {
                    n_children *= 3;
                }
            }
        }
        int idx_move = rand() % n_children;
        if (node_type == NodeType::FOOD_NODE) {
            int empty_food = 0;
            static std::array<bool, 77> used;
            for (int i = 0; i < 77; ++i) {
                used[i] = false;
            }
            for (int i = 0; i < v.boundary[4]; ++i) {
                used[v.geese[i].Id()] = true;
            }
            for (int i = 0; i < 2; ++i) {
                if (v.foods[i].Id() != -1) {
                    used[v.foods[i].Id()] = true;
                }
                else {
                    empty_food++;
                }
            }
            if (empty_food == 2 and 2 <= empty_cell) { // 2個選ぶ場合
                for (int i = 0; i < 77; ++i) {
                    if (used[i]) {
                        continue;
                    }
                    if (idx_move < empty_cell) {
                        v.foods[0] = Cpoint(i);
                        for (int j = i + 1; j < 77; ++j) {
                            if (used[j]) {
                                continue;
                            }
                            if (idx_move == 0) {
                                v.foods[1] = Cpoint(j);
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
                        if (v.foods[0].Id() == -1) {
                            v.foods[0] = Cpoint(i);
                        }
                        else {
                            v.foods[1] = Cpoint(i);
                        }
                        break;
                    }
                    idx_move--;
                }
            }
        }
        else {
            unsigned char agent_action = 0;
            for (int i = 0; i < 4; ++i) {
                if (v.goose_size((unsigned char) i) == 0) {
                    continue;
                }
                else {
                    unsigned char ith_idx_move = idx_move % 3;
                    unsigned char ith_idx_lastmove = 0;
                    if (v.last_actions & (1 << (i + i))) ith_idx_lastmove++;
                    if (v.last_actions & (1 << (i + i + 1))) ith_idx_lastmove+=2;
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
            v.Simulate(v, agent_action);
        }
    }
    for (int i = 0; i < 4; ++i) {
        // 生き残ってる
        if (v.boundary[i + 1] > v.boundary[i]) {
            int rank = 1;
            for (int j = 0; j < 4; ++j) {
                if (i == j) {
                    continue;
                }
                if (v.boundary[j + 1] - v.boundary[j] > v.boundary[i + 1] - v.boundary[i]) {
                    rank++;
                }
            }
            ranking[i] = rank;
        }
    }
    std::array<float ,4> res;
    for (int i = 0; i < 4; ++i) {
        res[i] = ranking[i];
    }
    return res;
}

//------------------------------------------------------------------------------
// 方向を指定して移動先を返す関数
Cpoint Duct2::Translate(Cpoint aPos, int Direction) {
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

