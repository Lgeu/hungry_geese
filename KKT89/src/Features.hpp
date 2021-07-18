#include <array>
#include <algorithm>
#include "Bitboard.hpp"

namespace feature {

enum struct Features {
    NEIGHBOR_UP_7,
    NEIGHBOR_DOWN_7,
    NEIGHBOR_LEFT_7,
    NEIGHBOR_RIGHT_7,
    LENGTH,
    DIFFERENCE_LENGTH_1ST,
    DIFFERENCE_LENGTH_2ND,
    DIFFERENCE_LENGTH_3RD,
    DIFFERENCE_LENGTH_4TH,
    RELATIVE_POSITION_TAIL,
    RELATIVE_POSITION_OPPONENT_HEAD,
    RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL,
    RELATIVE_POSITION_FOOD,
    MOVE_HISTORY,
    RELATIVE_POSITION_TAIL_ON_PLANE_X,
    RELATIVE_POSITION_TAIL_ON_PLANE_Y,
    N_REACHABLE_POSITIONS_WITHIN_1_STEP,  // 大きめに用意しておくと後で調整しやすそう？
    N_REACHABLE_POSITIONS_WITHIN_2_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_3_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_4_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_5_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_6_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_7_STEPS,
    N_REACHABLE_POSITIONS_WITHIN_8_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_2_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_3_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_4_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_5_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_6_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_7_STEPS,
    N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_8_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_2_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_3_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_4_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_5_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_6_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_7_STEPS,
    N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_8_STEPS,
    N_ALIVING_GEESE,
    N_OCCUPIED_POSITIONS,
    STEP,
    END  // 番兵的なやつ
};
constexpr auto N_FEATURES = (int)Features::END;

struct Min {
    std::array<int, N_FEATURES> data;
    constexpr Min() : data() {
        data[(int)Features::NEIGHBOR_UP_7] = 0;
        data[(int)Features::NEIGHBOR_DOWN_7] = 0;
        data[(int)Features::NEIGHBOR_LEFT_7] = 0;
        data[(int)Features::NEIGHBOR_RIGHT_7] = 0;
        data[(int)Features::LENGTH] = 1;
        data[(int)Features::DIFFERENCE_LENGTH_1ST] = -10;
        data[(int)Features::DIFFERENCE_LENGTH_2ND] = -10;
        data[(int)Features::DIFFERENCE_LENGTH_3RD] = -10;
        data[(int)Features::DIFFERENCE_LENGTH_4TH] = -10;
        data[(int)Features::RELATIVE_POSITION_TAIL] = 0;
        data[(int)Features::RELATIVE_POSITION_OPPONENT_HEAD] = 1;
        data[(int)Features::RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL] = 1;
        data[(int)Features::RELATIVE_POSITION_FOOD] = 1;
        data[(int)Features::MOVE_HISTORY] = 0;
        data[(int)Features::RELATIVE_POSITION_TAIL_ON_PLANE_X] = -30;
        data[(int)Features::RELATIVE_POSITION_TAIL_ON_PLANE_Y] = -30;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_1_STEP] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_2_STEPS] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_3_STEPS] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_4_STEPS] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_5_STEPS] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_6_STEPS] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_7_STEPS] = 0;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_8_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_2_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_3_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_4_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_5_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_6_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_7_STEPS] = 0;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_8_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_2_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_3_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_4_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_5_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_6_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_7_STEPS] = 0;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_8_STEPS] = 0;
        data[(int)Features::N_ALIVING_GEESE] = 2;
        data[(int)Features::N_OCCUPIED_POSITIONS] = 2;
        data[(int)Features::STEP] = 0;
    }
    const int& operator[](const Features& feature) const {
        return data[(int)feature];
    }
};

struct Max {
    std::array<int, N_FEATURES> data;
    constexpr Max() : data() {
        data[(int)Features::NEIGHBOR_UP_7] = (1 << 7) - 1;
        data[(int)Features::NEIGHBOR_DOWN_7] = (1 << 7) - 1;
        data[(int)Features::NEIGHBOR_LEFT_7] = (1 << 7) - 1;
        data[(int)Features::NEIGHBOR_RIGHT_7] = (1 << 7) - 1;
        data[(int)Features::LENGTH] = 77;
        data[(int)Features::DIFFERENCE_LENGTH_1ST] = 10;
        data[(int)Features::DIFFERENCE_LENGTH_2ND] = 10;
        data[(int)Features::DIFFERENCE_LENGTH_3RD] = 10;
        data[(int)Features::DIFFERENCE_LENGTH_4TH] = 10;
        data[(int)Features::RELATIVE_POSITION_TAIL] = 76;
        data[(int)Features::RELATIVE_POSITION_OPPONENT_HEAD] = 76;
        data[(int)Features::RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL] = 76;
        data[(int)Features::RELATIVE_POSITION_FOOD] = 76;
        data[(int)Features::MOVE_HISTORY] = 4 * 4 * 4 * 4 - 1;
        data[(int)Features::RELATIVE_POSITION_TAIL_ON_PLANE_X] = 30;
        data[(int)Features::RELATIVE_POSITION_TAIL_ON_PLANE_Y] = 30;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_1_STEP] = 5;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_2_STEPS] = 13;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_3_STEPS] = 25;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_4_STEPS] = 39;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_5_STEPS] = 53;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_6_STEPS] = 65;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_7_STEPS] = 73;
        data[(int)Features::N_REACHABLE_POSITIONS_WITHIN_8_STEPS] = 77;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_2_STEPS] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_3_STEPS] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_4_STEPS] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_5_STEPS] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_6_STEPS] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_7_STEPS] = 3;
        data[(int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_8_STEPS] = 3;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP] = 5;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_2_STEPS] = 13;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_3_STEPS] = 25;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_4_STEPS] = 39;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_5_STEPS] = 53;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_6_STEPS] = 65;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_7_STEPS] = 73;
        data[(int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_8_STEPS] = 77;
        data[(int)Features::N_ALIVING_GEESE] = 4;
        data[(int)Features::N_OCCUPIED_POSITIONS] = 77;
        data[(int)Features::STEP] = 198;
    }
    const int& operator[](const Features& feature) const {
        return data[(int)feature];
    }
};

struct Offset {
    std::array<int, N_FEATURES+1> data;
    constexpr Offset(const Min& min_, const Max& max_) : data() {
        int idx = 0;
        for (auto i = 0; i < N_FEATURES; i++) {
            data[i] = idx - min_.data[i];
            idx += max_.data[i] - min_.data[i] + 1;
        }
        data[N_FEATURES] = idx;
    }
    const int& operator[](const Features& feature) const {
        return data[(int)feature];
    }
};

constexpr auto MIN = Min();
constexpr auto MAX = Max();
constexpr auto OFFSET = Offset(MIN, MAX);
constexpr auto NN_INPUT_DIM = OFFSET.data[N_FEATURES];

void PrintFeatureBoundary() {
    // 特徴量カテゴリの境界位置と中心位置を出力
    int idx = 0;
    std::cout << "BOUNDARY = [";
    for (auto i = 0; i < N_FEATURES; i++) {
        std::cout << idx << ",";
        idx += MAX.data[i] - MIN.data[i] + 1;
    }
    std::cout << idx << "]" << std::endl;

    std::cout << "OFFSET = [";
    for (auto i = 0; i < N_FEATURES; i++) {
        std::cout << OFFSET.data[i] << ",";
    }
    std::cout << "]" << std::endl;
}

constexpr auto MAX_FEATURE_REACHABLE_CALCULATION = 8;

template<class IntStack1, class IntStack2>  // std::vector とだいたい同等の操作ができるクラス
void ExtractFeatures(
    const std::array<IntStack1, 4>& geese,
    const std::array<int, 2>& foods,
    const int& current_step,
    std::array<IntStack2, 4>& agent_features,    // output
    IntStack2& condition_features           // output
) {
    // いずれかの geese がいる位置が 1、それ以外が 0  // 差分計算可能
    auto occupied_bitboard = BitBoard();
    for (const auto& goose : geese) {
        for (const auto& p : goose) {
            occupied_bitboard.Flip(p);
            ASSERT(occupied_bitboard[p], "このビットは立ってるはずだよ");
        }
    }

    using nagiss_library::Vec2;
    using nagiss_library::clipped;
    // この周辺を効率化する方法はいろいろ考えられるけど、加算回数が減るだけなのでどれくらいいい感じになるのか不明
    // 差分計算（親ノード、類似ノード）
    // ベクトル量子化

    // 初期化
    for (int i = 0; i < 4; i++) agent_features[i].clear();
    condition_features.clear();

    // 前処理: ソートした長さ
    auto sorted_lengths = std::array<int, 4>();
    for (int i = 0; i < 4; i++) sorted_lengths[i] = geese[i].size();
    std::sort(sorted_lengths.begin(), sorted_lengths.end(), std::greater<>());

    // 前処理: future ステップ以内に到達可能な場所 (他 geese の頭が動かないと仮定)  // 正しくない場合もある (長さが 1 のときなど)
    auto not_occupied = BitBoard(occupied_bitboard);
    not_occupied.Invert();
    auto reachable_positions = std::array<std::array<BitBoard, MAX_FEATURE_REACHABLE_CALCULATION + 1>, 4>();  // 各 goose の 1 ~ 8 ステップ後に到達可能な場所
    for (int i = 0; i < 4; i++) {
        if (geese[i].size() == 0) continue;
        reachable_positions[i][0].Flip(geese[i].front());
    }
    //cout << "occupied_bitboard" << endl;
    //occupied_bitboard.Print();
    for (int future = 1, clearing_count = 1; future <= MAX_FEATURE_REACHABLE_CALCULATION; future++, clearing_count++) {
        // 短くなる処理
        for (int i = 0; i < 4; i++) {
            auto geese_i_clearing_idx = geese[i].size() - clearing_count;
            if (geese_i_clearing_idx < 0) continue;
            const auto& idx = geese[i][geese_i_clearing_idx];
            not_occupied.Flip(idx);
            ASSERT(not_occupied[idx], "しっぽを消そうと思ったらしっぽが無かったよ");
        }
        // もう一回短くなる
        if ((current_step + future) % 40 == 0) {  // この条件合ってる？要確認
            clearing_count++;
            for (int i = 0; i < 4; i++) {
                auto geese_i_clearing_idx = geese[i].size() - clearing_count;
                if (geese_i_clearing_idx < 0) continue;
                const auto& idx = geese[i][geese_i_clearing_idx];
                not_occupied.Flip(idx);
                ASSERT(not_occupied[idx], "しっぽを消そうと思ったらしっぽが無かったよ");
            }
        }
        for (int i = 0; i < 4; i++) {
            if (geese[i].size() == 0) continue;
            const auto& prev_reachable_positions = reachable_positions[i][future - 1];
            auto& next_reachable_positions = reachable_positions[i][future];
            next_reachable_positions = prev_reachable_positions;//  if (future == 1) { cout << "prev" << endl; prev_reachable_positions.Print(); }
            auto tmp = prev_reachable_positions;  tmp.ShiftRight();  next_reachable_positions |= tmp;//  if (future == 1) { cout << "R" << endl; tmp.Print(); }
            tmp = prev_reachable_positions;  tmp.ShiftLeft();   next_reachable_positions |= tmp;//  if (future == 1) { cout << "L" << endl; tmp.Print(); }
            tmp = prev_reachable_positions;  tmp.ShiftDown();   next_reachable_positions |= tmp;//  if (future == 1) { cout << "D" << endl; tmp.Print(); }
            tmp = prev_reachable_positions;  tmp.ShiftUp();     next_reachable_positions |= tmp;//  if (future == 1) { cout << "U" << endl; tmp.Print(); }
            next_reachable_positions &= not_occupied;
        }
    }

    // 各エージェント視点の特徴量
    for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
        const auto& goose = geese[idx_agents];
        if (goose.size() == 0) continue;  // もう脱落していた場合、飛ばす

        // 基本的な情報を抽出
        const auto& head = goose.front();
        const auto& tail = goose.back();
        auto& features = agent_features[idx_agents];
        const auto head_vec = Vec2<int>(head / 11, head % 11);
        const auto tail_vec = Vec2<int>(tail / 11, tail % 11);

        // 上下左右の近傍 7 マス
        const auto neighbor_up_7 = occupied_bitboard.NeighborUp7(head);
        ASSERT_RANGE(neighbor_up_7, MIN[Features::NEIGHBOR_UP_7], MAX[Features::NEIGHBOR_UP_7] + 1);
        features.push(OFFSET[Features::NEIGHBOR_UP_7] + neighbor_up_7);
        const auto neighbor_down_7 = occupied_bitboard.NeighborDown7(head);
        ASSERT_RANGE(neighbor_down_7, MIN[Features::NEIGHBOR_DOWN_7], MAX[Features::NEIGHBOR_DOWN_7] + 1);
        features.push(OFFSET[Features::NEIGHBOR_DOWN_7] + neighbor_down_7);
        const auto neighbor_left_7 = occupied_bitboard.NeighborLeft7(head);
        ASSERT_RANGE(neighbor_left_7, MIN[Features::NEIGHBOR_LEFT_7], MAX[Features::NEIGHBOR_LEFT_7] + 1);
        features.push(OFFSET[Features::NEIGHBOR_LEFT_7] + neighbor_left_7);
        const auto neighbor_right_7 = occupied_bitboard.NeighborRight7(head);
        ASSERT_RANGE(neighbor_right_7, MIN[Features::NEIGHBOR_RIGHT_7], MAX[Features::NEIGHBOR_RIGHT_7] + 1);
        features.push(OFFSET[Features::NEIGHBOR_RIGHT_7] + neighbor_right_7);

        // goose の長さ
        const auto length = goose.size();
        ASSERT_RANGE(length, MIN[Features::LENGTH], MAX[Features::LENGTH] + 1);
        features.push(OFFSET[Features::LENGTH] + length);

        // [1-4] 番目に長い goose との長さの差
        for (int rank = 0; rank < 4; rank++) {
            auto difference_length = length - sorted_lengths[rank];
            difference_length = clipped(difference_length, MIN[(Features)(rank + (int)Features::DIFFERENCE_LENGTH_1ST)], MAX[(Features)(rank + (int)Features::DIFFERENCE_LENGTH_1ST)]);
            features.push(OFFSET[(Features)(rank + (int)Features::DIFFERENCE_LENGTH_1ST)] + difference_length);
        }

        // しっぽの位置
        auto CalcRelativePosition = [](const int& s, const int& t) {
            // s から見た t の位置  // 遅かったら表引きに書き換える
            const auto s_vec = Vec2<int>(s / 11, s % 11);
            const auto t_vec = Vec2<int>(t / 11, t % 11);
            auto relative_positioin_vec = t_vec - s_vec;
            if (relative_positioin_vec.y < 0) relative_positioin_vec.y += 7;
            if (relative_positioin_vec.x < 0) relative_positioin_vec.x += 11;
            ASSERT_RANGE(relative_positioin_vec.y, 0, 7);
            ASSERT_RANGE(relative_positioin_vec.x, 0, 11);
            const auto relative_position = relative_positioin_vec.y * 11 + relative_positioin_vec.x;
            return relative_position;
        };
        const auto relative_tail = CalcRelativePosition(head, tail);
        ASSERT_RANGE(relative_tail, MIN[Features::RELATIVE_POSITION_TAIL], MAX[Features::RELATIVE_POSITION_TAIL] + 1);  // 長さが 1 の場合に頭と同じ (0) になる
        features.push(OFFSET[Features::RELATIVE_POSITION_TAIL] + relative_tail);

        // 敵の頭の位置・しっぽから見た敵の頭の位置
        for (int opponent = 0; opponent < 4; opponent++) {
            if (opponent == idx_agents || geese[opponent].size() == 0) continue;
            const auto& opponent_head = geese[opponent].front();
            const auto relative_opponent_head = CalcRelativePosition(head, opponent_head);
            ASSERT_RANGE(relative_opponent_head, MIN[Features::RELATIVE_POSITION_OPPONENT_HEAD], MAX[Features::RELATIVE_POSITION_OPPONENT_HEAD] + 1);
            features.push(OFFSET[Features::RELATIVE_POSITION_OPPONENT_HEAD] + relative_opponent_head);

            const auto relative_opponent_head_from_tail = CalcRelativePosition(tail, opponent_head);
            ASSERT_RANGE(relative_opponent_head_from_tail, MIN[Features::RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL], MAX[Features::RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL] + 1);
            features.push(OFFSET[Features::RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL] + relative_opponent_head_from_tail);
        }

        // 食べ物の位置
        for (int idx_foods = 0; idx_foods < foods.size(); idx_foods++) {
            const auto relative_food = CalcRelativePosition(head, foods[idx_foods]);
            ASSERT_RANGE(relative_food, MIN[Features::RELATIVE_POSITION_FOOD], MAX[Features::RELATIVE_POSITION_FOOD] + 1);
            features.push(OFFSET[Features::RELATIVE_POSITION_FOOD] + relative_food);
        }

        // 行動履歴
        auto history = 0;
        auto direction = 0;
        for (int i = 0; i < 4; i++) {
            if (goose.size() <= i + 1) {
                direction ^= 2;  // goose が短い場合は前回と逆向きに入れる
            }
            else {
                const auto relative_position = CalcRelativePosition(goose[i + 1], goose[i]);
                switch (relative_position) {
                case 66:
                    direction = 0;
                    break;
                case 1:
                    direction = 1;
                    break;
                case 11:
                    direction = 2;
                    break;
                case 10:
                    direction = 3;
                    break;
                default:
                    ASSERT(false, "何か間違ってるよ");
                }
            }
            history |= direction << i * 2;
        }
        ASSERT_RANGE(history, MIN[Features::MOVE_HISTORY], MAX[Features::MOVE_HISTORY] + 1);
        features.push(OFFSET[Features::MOVE_HISTORY] + history);

        // 平面上に置いたときのしっぽの相対位置  // 差分計算可能
        auto relative_tail_on_plane_x = 0;
        auto relative_tail_on_plane_y = 0;
        auto old_yx = Vec2<int>(head / 11, head % 11);
        for (int i = 0; i < goose.size() - 1; i++) {  // 3 つずつずらした方が効率的だけど面倒！
            const auto relative_position = CalcRelativePosition(goose[i + 1], goose[i]);
            switch (relative_position) {
            case 66:
                relative_tail_on_plane_y--;
                break;
            case 1:
                relative_tail_on_plane_x++;
                break;
            case 11:
                relative_tail_on_plane_y++;
                break;
            case 10:
                relative_tail_on_plane_x--;
                break;
            default:
                ASSERT(false, "何か間違ってるよ");
            }
        }
        relative_tail_on_plane_x = clipped(relative_tail_on_plane_x, MIN[Features::RELATIVE_POSITION_TAIL_ON_PLANE_X], MAX[Features::RELATIVE_POSITION_TAIL_ON_PLANE_X]);
        relative_tail_on_plane_y = clipped(relative_tail_on_plane_y, MIN[Features::RELATIVE_POSITION_TAIL_ON_PLANE_Y], MAX[Features::RELATIVE_POSITION_TAIL_ON_PLANE_Y]);
        features.push(OFFSET[Features::RELATIVE_POSITION_TAIL_ON_PLANE_X] + relative_tail_on_plane_x);
        features.push(OFFSET[Features::RELATIVE_POSITION_TAIL_ON_PLANE_Y] + relative_tail_on_plane_y);

        // n ステップ以内に到達可能な場所の数 (他 geese の頭が動かないと仮定)
        for (int n = 1; n <= MAX_FEATURE_REACHABLE_CALCULATION; n++) {
            auto n_reachable_positions_within_n_steps = reachable_positions[idx_agents][n].Popcount();
            ASSERT_RANGE(
                n_reachable_positions_within_n_steps,
                MIN[(Features)(n - 1 + (int)Features::N_REACHABLE_POSITIONS_WITHIN_1_STEP)],
                MAX[(Features)(n - 1 + (int)Features::N_REACHABLE_POSITIONS_WITHIN_1_STEP)] + 1
            );
            features.push(OFFSET[(Features)(n - 1 + (int)Features::N_REACHABLE_POSITIONS_WITHIN_1_STEP)] + n_reachable_positions_within_n_steps);
        }

        // n ステップ以内に到達可能な場所が被ってる敵の数
        for (int n = 1; n <= MAX_FEATURE_REACHABLE_CALCULATION; n++) {
            auto n_opponents_sharing_reachable_positions_within_n_steps = 0;
            for (int i = 0; i < 4; i++) {
                if (i == idx_agents || geese[i].size() == 0) continue;
                n_opponents_sharing_reachable_positions_within_n_steps += (int)!(reachable_positions[idx_agents][n] & reachable_positions[i][n]).Empty();
            }
            ASSERT_RANGE(
                n_opponents_sharing_reachable_positions_within_n_steps,
                MIN[(Features)(n - 1 + (int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP)],
                MAX[(Features)(n - 1 + (int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP)] + 1
            );
            features.push(OFFSET[(Features)(n - 1 + (int)Features::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_1_STEP)] + n_opponents_sharing_reachable_positions_within_n_steps);
        }

        // n ステップ以内に自分だけが到達可能な場所の数
        for (int n = 1; n <= MAX_FEATURE_REACHABLE_CALCULATION; n++) {
            auto not_opponents_reachable = BitBoard();
            for (int i = 0; i < 4; i++) {
                if (i == idx_agents || geese[i].size() == 0) continue;
                not_opponents_reachable |= reachable_positions[i][n];
            }
            not_opponents_reachable.Invert();
            auto n_exclusively_reachable_positions_within_n_steps = (reachable_positions[idx_agents][n] & not_opponents_reachable).Popcount();
            ASSERT_RANGE(
                n_exclusively_reachable_positions_within_n_steps,
                MIN[(Features)(n - 1 + (int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP)],
                MAX[(Features)(n - 1 + (int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP)] + 1
            );
            features.push(OFFSET[(Features)(n - 1 + (int)Features::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_1_STEP)] + n_exclusively_reachable_positions_within_n_steps);
        }
    }

    // 全体の特徴量
    // TODO
    // 生存人数・埋まってるマスの数
    auto n_aliving_geese = 0;
    auto n_occupied_positions = 0;
    for (int i = 0; i < 4; i++) {
        if (geese[i].size() == 0) continue;
        n_aliving_geese++;
        n_occupied_positions += geese[i].size();
    }
    ASSERT_RANGE(n_aliving_geese, MIN[Features::N_ALIVING_GEESE], MAX[Features::N_ALIVING_GEESE] + 1);
    condition_features.push(OFFSET[Features::N_ALIVING_GEESE] + n_aliving_geese);
    ASSERT_RANGE(n_occupied_positions, MIN[Features::N_OCCUPIED_POSITIONS], MAX[Features::N_OCCUPIED_POSITIONS] + 1);
    condition_features.push(OFFSET[Features::N_OCCUPIED_POSITIONS] + n_occupied_positions);

    // ステップ
    ASSERT_RANGE(current_step, MIN[Features::STEP], MAX[Features::STEP] + 1);
    condition_features.push(OFFSET[Features::STEP] + current_step);
}

}
