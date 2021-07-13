#pragma once
#include "GreedyAgent.hpp"

#include "library.hpp"

namespace hungry_geese {


namespace evaluation_function {

using namespace std;

template<class T, int dim1, int dim2>
struct Matrix {
    array<array<T, dim2>, dim1> data;
    array<T, dim2>& operator[](const int& idx) {
        return data[idx];
    }
    const array<T, dim2>& operator[](const int& idx) const {
        return data[idx];
    }
    array<T, dim1* dim2>& Ravel() {
        union U {
            array<array<T, dim2>, dim1> data;
            array<T, dim1* dim2> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const array<T, dim1* dim2>& Ravel() const {
        union U {
            array<array<T, dim2>, dim1> data;
            array<T, dim1* dim2> raveled;
        };
        return ((U*)&data)->raveled;
    }
    void Fill(const T& fill_value) {
        fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    inline auto& operator+=(const Matrix& rhs) {
        for (int i = 0; i < dim1 * dim2; i++) Ravel()[i] += rhs.Ravel()[i];
        return *this;
    }
    void Print() const {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                cout << data[i][j] << " \n"[j == dim2 - 1];
            }
        }
    }
};

template<class T, int dim1, int dim2, int dim3>
struct Tensor3 {
    array<Matrix<T, dim2, dim3>, dim1> data;
    Matrix<T, dim2, dim3>& operator[](const int& idx) {
        return data[idx];
    }
    const Matrix<T, dim2, dim3>& operator[](const int& idx) const {
        return data[idx];
    }
    array<T, dim1* dim2* dim3>& Ravel() {
        union U {
            array<Matrix<T, dim2, dim3>, dim1> data;
            array<T, dim1* dim2* dim3> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const array<T, dim1* dim2* dim3>& Ravel() const {
        union U {
            array<Matrix<T, dim2, dim3>, dim1> data;
            array<T, dim1* dim2* dim3> raveled;
        };
        return ((U*)&data)->raveled;
    }
    template<int new_dim_1, int new_dim_2>
    Matrix<T, new_dim_1, new_dim_2>  View() {
        static_assert(dim1 * dim2 * dim3 == new_dim_1 * new_dim_2, "View の次元がおかしいよ");
        union U {
            Tensor3 data;
            Matrix<new_dim_1, new_dim_2> view;
        };
        return ((U*)this)->view;
    }
    void Fill(const T& fill_value) {
        fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    template<int p0, int p1, int p2>
    auto Permute() const {
        // d0 が 1 の場合、もともと 1 次元目だったところが 0 次元目に来る
        constexpr static auto d = array<int, 3>{dim1, dim2, dim3};
        auto permuted = Tensor3<T, d[p0], d[p1], d[p2]>();
        for (int i1 = 0; i1 < dim1; i1++)
            for (int i2 = 0; i2 < dim2; i2++)
                for (int i3 = 0; i3 < dim3; i3++)
                    permuted
                    [array<int, 3>{i1, i2, i3} [p0] ]
        [array<int, 3>{i1, i2, i3} [p1] ]
        [array<int, 3>{i1, i2, i3} [p2] ] = data[i1][i2][i3];
        return permuted;
    }
    inline auto& operator+=(const Tensor3& rhs) {
        for (int i = 0; i < dim1 * dim2 * dim3; i++) Ravel()[i] += rhs.Ravel()[i];
        return *this;
    }
    void Print() const {
        for (int i = 0; i < dim1; i++) {
            data[i].Print();
            if (i != dim1 - 1) cout << endl;
        }
    }
};

template<class T, int dim1, int dim2, int dim3, int dim4>
struct Tensor4 {
    array<Tensor3<T, dim2, dim3, dim4>, dim1> data;

    Tensor3<T, dim2, dim3, dim4>& operator[](const int& idx) {
        return data[idx];
    }
    const Tensor3<T, dim2, dim3, dim4>& operator[](const int& idx) const {
        return data[idx];
    }
    array<T, dim1* dim2* dim3* dim4>& Ravel() {
        union U {
            array<Tensor3<T, dim2, dim3, dim4>, dim1> data;
            array<T, dim1* dim2* dim3* dim4> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const array<T, dim1* dim2* dim3* dim4>& Ravel() const {
        union U {
            array<Tensor3<T, dim2, dim3, dim4>, dim1> data;
            array<T, dim1* dim2* dim3* dim4> raveled;
        };
        return ((U*)&data)->raveled;
    }
    void Fill(const T& fill_value) {
        fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    template<int p0, int p1, int p2, int p3>
    auto Permute() const {
        // d0 が 1 の場合、もともと 1 次元目だったところが 0 次元目に来る
        constexpr static auto d = array<int, 4>{dim1, dim2, dim3, dim4};
        auto permuted = Tensor4<T, d[p0], d[p1], d[p2], d[p3]>();
        for (int i1 = 0; i1 < dim1; i1++)
            for (int i2 = 0; i2 < dim2; i2++)
                for (int i3 = 0; i3 < dim3; i3++)
                    for (int i4 = 0; i4 < dim4; i4++)
                        permuted
                        [array<int, 4>{i1, i2, i3, i4} [p0] ]
        [array<int, 4>{i1, i2, i3, i4} [p1] ]
        [array<int, 4>{i1, i2, i3, i4} [p2] ]
        [array<int, 4>{i1, i2, i3, i4} [p3] ] = data[i1][i2][i3][i4];
        return permuted;
    }
    inline auto& operator+=(const Tensor4& rhs) {
        for (int i = 0; i < dim1 * dim2 * dim3 * dim4; i++) Ravel()[i] += rhs.Ravel()[i];
        return *this;
    }
    void Print() const {
        for (int i = 0; i < dim1; i++) {
            data[i].Print();
            if (i != dim1 - 1) cout << endl << endl;
        }
    }
};

namespace F {
    template<class Tensor>
    void Relu_(Tensor& input) {
        for (auto&& value : input.Ravel()) {
            value = max(value, (typename remove_reference<decltype(input.Ravel()[0])>::type)0);
        }
    }
    template<typename T, unsigned siz>
    void Relu_(array<T, siz>& input) {
        for (auto&& value : input) {
            value = max(value, (T)0);
        }
    }
    template<class T, size_t siz>
    void Softmax_(array<T, siz>& input) {
        auto ma = numeric_limits<float>::min();
        for (const auto& v : input) if (ma < v) ma = v;
        auto s = 0.0f;
        for (const auto& v : input) s += expf(v - ma);
        auto c = ma + logf(s);
        for (auto&& v : input) v = expf(v - c);
    }
}

template<int n_features>
struct BatchNorm2d {
    struct Parameter {
        array<float, n_features> weight;  // gamma
        array<float, n_features> bias;    // beta
        array<float, n_features> running_mean;
        array<float, n_features> running_var;
        Parameter() : weight(), bias(), running_mean(), running_var() {
            // weight と running_var は 1 で初期化
            fill(weight.begin(), weight.end(), 1.0f);
            fill(running_var.begin(), running_var.end(), 1.0f);
        }
    } parameters;

    // コンストラクタ
    BatchNorm2d() : parameters() {}

    template<int height, int width>
    void Forward_(Tensor3<float, n_features, height, width>& input) const {
        for (int channel = 0; channel < n_features; channel++) {
            const auto coef = parameters.weight[channel] / sqrtf(parameters.running_var[channel] + 1e-5f);
            const auto bias_ = parameters.bias[channel] - coef * parameters.running_mean[channel];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    input[channel][y][x] = input[channel][y][x] * coef + bias_;
                }
            }
        }
    }
};

template<int input_dim, int output_dim, int kernel_size = 3>
struct TorusConv2d {
    struct Parameter {
        Tensor4<float, output_dim, input_dim, kernel_size, kernel_size> weight;
        array<float, output_dim> bias;
    } parameters;

    BatchNorm2d<output_dim> batchnorm;


    // コンストラクタ
    TorusConv2d() : parameters(), batchnorm() {}

    constexpr void SetParameters() {
        // 下手に実装すると拡張しにくくなりそう、型が混ざってる場合の対応が厄介
    }

    template<int height, int width>
    void Forward(
        const Tensor3<float, input_dim, height, width>& input,
        Tensor3<float, output_dim, height, width>& output) const {
        // これループの順番変えるだけで速度だいぶ変わりそう…
        constexpr auto pad = kernel_size / 2;
        static_assert(height >= pad, "2 周回るようなカーネルサイズは未対応だよ");
        static_assert(width >= pad, "2 周回るようなカーネルサイズは未対応だよ");
        // 効率化した版
        const auto permuted_input = input.template Permute<1, 2, 0>();
        const auto permuted_weight = parameters.weight.template Permute<2, 3, 0, 1>();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int out_channel = 0; out_channel < output_dim; out_channel++) {
                    output[out_channel][y][x] = parameters.bias[out_channel];
                }
                for (int ky = 0; ky < kernel_size; ky++) {
                    auto y_from = y + ky - pad;
                    if (y_from < 0) y_from += height;
                    else if (y_from >= height) y_from -= height;
                    for (int kx = 0; kx < kernel_size; kx++) {
                        auto x_from = x + kx - pad;
                        if (x_from < 0) x_from += width;
                        else if (x_from >= width) x_from -= width;
                        for (int out_channel = 0; out_channel < output_dim; out_channel++) {
                            for (int in_channel = 0; in_channel < input_dim; in_channel++) {
                                output[out_channel][y][x] += permuted_input[y_from][x_from][in_channel] * permuted_weight[ky][kx][out_channel][in_channel];
                            }
                        }
                    }
                }
            }
        }
        batchnorm.Forward_(output);
    }
};

template<int in_features, int out_features>
struct Linear {
    struct Parameters {
        Matrix<float, out_features, in_features> weight;
        array<float, out_features> bias;
    } parameters;

    // コンストラクタ
    Linear() : parameters() {}

    void Forward(const array<float, in_features>& input, array<float, out_features>& output) const {
        output = parameters.bias;
        for (int out_channel = 0; out_channel < out_features; out_channel++) {
            for (int in_channel = 0; in_channel < in_features; in_channel++) {
                output[out_channel] += input[in_channel] * parameters.weight[out_channel][in_channel];
            }
        }
    }
};

template<int num_embeddings, int embedding_dim>
struct EmbeddingBag {
    // mode="sum" のみ対応
    struct Parameters {
        Matrix<float, num_embeddings, embedding_dim> weight;
    } parameters;

    // コンストラクタ
    EmbeddingBag() : parameters() {}

    template<class Vector>  // Stack<int, n> とか
    void Forward(const Vector& input, array<float, embedding_dim>& output) const {
        fill(output.begin(), output.end(), 0.0f);
        for (const auto& idx : input) {
            for (int dim = 0; dim < embedding_dim; dim++) {
                output[dim] += parameters.weight[idx][dim];
            }
        }
    }
};

template<int in_dim, int out_dim = 5, int hidden_1 = 256, int hidden_2 = 32>
struct Model {
    // TODO: Embed を 1 つに統一
    EmbeddingBag<in_dim, hidden_1> embed;
    Linear<hidden_1, hidden_2> linear_condition;
    Linear<hidden_1, hidden_2> linear_2;
    Linear<hidden_2, hidden_2> linear_3;
    Linear<hidden_2, out_dim> linear_4;

    // コンストラクタ
    Model() : embed(), linear_condition(), linear_2(), linear_3(), linear_4() {}

    template<class Vector1, class Vector2>  // Stack<int, n> とか
    void Forward(const array<Vector1, 4>& agent_features, const Vector2& condition_features, Matrix<float, 4, out_dim>& output) {
        // (1)
        static auto agent_embedded = Matrix<float, 4, hidden_1>();
        for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
            embed.Forward(agent_features[idx_agents], agent_embedded[idx_agents]);
        }
        F::Relu_(agent_embedded);

        // (2)
        static auto condition_embedded = array<float, hidden_1>();
        embed.Forward(condition_features, condition_embedded);
        F::Relu_(condition_embedded);
        for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
            for (int dim = 0; dim < hidden_1; dim++) {
                condition_embedded[dim] += agent_embedded[idx_agents][dim];  // Vector 構造体を作るべきだった感
            }
        }

        // (3)
        static auto condition_hidden = array<float, hidden_2>();
        linear_condition.Forward(condition_embedded, condition_hidden);
        F::Relu_(condition_hidden);

        // (4)
        static auto hidden_state_2 = Matrix<float, 4, hidden_2>();
        for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
            linear_2.Forward(agent_embedded[idx_agents], hidden_state_2[idx_agents]);
        }
        F::Relu_(hidden_state_2);
        for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
            for (int dim = 0; dim < hidden_2; dim++) {
                hidden_state_2[idx_agents][dim] += condition_hidden[dim];
            }
        }

        // (5)
        static auto hidden_state_3 = Matrix<float, 4, hidden_2>();
        for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
            linear_3.Forward(hidden_state_2[idx_agents], hidden_state_3[idx_agents]);
        }
        F::Relu_(hidden_state_3);

        // (6)
        for (int idx_agents = 0; idx_agents < 4; idx_agents++) {
            linear_4.Forward(hidden_state_3[idx_agents], output[idx_agents]);
        }
    }

};


struct BitBoard {
    using ull = unsigned long long;
    ull lo, hi;
    static constexpr ull mask_right_lo = 0b10000000000'10000000000'10000000000'10000000000'10000000000ull, mask_right_hi = 0b10000000000'10000000000ull;
    static constexpr ull mask_left_lo =  0b00000000001'00000000001'00000000001'00000000001'00000000001ull, mask_left_hi =  0b00000000001'00000000001ull;
    static constexpr ull mask_down_lo =  0b11111111111'00000000000'00000000000'00000000000'00000000000ull, mask_down_hi =  0b11111111111'00000000000ull;
    static constexpr ull mask_up_lo =    0b00000000000'00000000000'00000000000'00000000000'11111111111ull, mask_up_hi =    0b00000000000'11111111111ull;
    static constexpr ull mask_all_lo =   0b11111111111'11111111111'11111111111'11111111111'11111111111ull, mask_all_hi =   0b11111111111'11111111111ull;
    inline BitBoard() : lo(), hi() {}
    inline BitBoard(const int& idx) : lo(), hi() { Flip(idx); }
    inline void Print() const {
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 11; x++) {
                cout << ((lo >> y * 11 + x) & 1);
            }
            cout << endl;
        }
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 11; x++) {
                cout << ((hi >> y * 11 + x) & 1);
            }
            cout << endl;
        }
    }
    inline void Flip(const int& idx) {
        if (idx < 55) lo ^= 1ull << idx;
        else hi ^= 1ull << idx - 55;
    }
    inline void ShiftRight() {
        const auto masked_lo = lo & mask_right_lo;  // 右端を取り出す
        const auto masked_hi = hi & mask_right_hi;
        lo ^= masked_lo;                            // 右端を消す
        hi ^= masked_hi;
        lo <<= 1;                                   // 右にシフト
        hi <<= 1;
        lo |= masked_lo >> 10;                     // 右端にあったものを左端に持ってくる
        hi |= masked_hi >> 10;
        ASSERT_RANGE(lo, 0ull, 1ull << 55);
        ASSERT_RANGE(hi, 0ull, 1ull << 22);
    }
    inline void ShiftLeft() {
        const auto masked_lo = lo & mask_left_lo;
        const auto masked_hi = hi & mask_left_hi;
        lo ^= masked_lo;
        hi ^= masked_hi;
        lo >>= 1;
        hi >>= 1;
        lo |= masked_lo << 10;
        hi |= masked_hi << 10;
        ASSERT_RANGE(lo, 0ull, 1ull << 55);
        ASSERT_RANGE(hi, 0ull, 1ull << 22);
    }
    inline void ShiftDown() {
        const auto masked_lo = lo & mask_down_lo;
        const auto masked_hi = hi & mask_down_hi;
        lo ^= masked_lo;
        hi ^= masked_hi;
        lo <<= 11;
        hi <<= 11;
        lo |= masked_hi >> 11;
        hi |= masked_lo >> 44;
        ASSERT_RANGE(lo, 0ull, 1ull << 55);
        ASSERT_RANGE(hi, 0ull, 1ull << 22);
    }
    inline void ShiftUp() {
        const auto masked_lo = lo & mask_up_lo;
        const auto masked_hi = hi & mask_up_hi;
        lo >>= 11;
        hi >>= 11;
        lo |= masked_hi << 44;
        hi |= masked_lo << 11;
        ASSERT_RANGE(lo, 0ull, 1ull << 55);
        ASSERT_RANGE(hi, 0ull, 1ull << 22);
    }
    inline void Invert() {
        lo ^= mask_all_lo;
        hi ^= mask_all_hi;
    }
    inline int Popcount() const {
        using nagiss_library::popcount;
        return popcount(lo) + popcount(hi);
    }
    template<unsigned size>
    inline int Neighbor(const int& idx, const array<nagiss_library::Vec2<int>, size>& dyxs) const {
        int res = 0;
        auto yx0 = nagiss_library::Vec2<int>{ idx / 11, idx % 11 };
        for (int i = 0; i < (int)size; i++) {
            auto yx = yx0 + dyxs[i];
            if (yx.y < 0) yx.y += 7;
            else if (yx.y >= 7) yx.y -= 7;
            if (yx.x < 0) yx.x += 11;
            else if (yx.x >= 11) yx.x -= 11;
            auto idx_ = yx.y * 11 + yx.x;
            res |= ((idx_ < 55 ? lo >> idx_ : hi >> idx_ - 55) & 1) << i;
        }
        return res;
    }
    inline int Neighbor12(const int& idx) const {
        // 00100
        // 01110
        // 11@11
        // 01110
        // 00100
        // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = array<Vec2<int>, 12>{Vec2<int>{-2, 0}, { -1,-1 }, { -1,0 }, { -1,1 }, { 0,-2 }, { 0,-1 }, { 0,1 }, { 0,2 }, { 1,-1 }, { 1,0 }, { 1,1 }, { 2,0 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborUp7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = array<Vec2<int>, 7>{Vec2<int>{-3, 0}, { -2,-1 }, { -2,0 }, { -2,1 }, { -1,-1 }, { -1,0 }, { -1,1 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborDown7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = array<Vec2<int>, 7>{Vec2<int>{3, 0}, { 2,-1 }, { 2,0 }, { 2,1 }, { 1,-1 }, { 1,0 }, { 1,1 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborLeft7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = array<Vec2<int>, 7>{Vec2<int>{0, -3}, { -1,-2 }, { 0,-2 }, { 1,-2 }, { -1,-1 }, { 0,-1 }, { 1,-1 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborRight7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = array<Vec2<int>, 7>{Vec2<int>{0, 3}, { -1,2 }, { 0,2 }, { 1,2 }, { -1,1 }, { 0,1 }, { 1,1 }};
        return Neighbor(idx, dyxs);
    }
    inline bool Empty() const {
        return lo == 0ull && hi == 0ull;
    }
    inline BitBoard& operator&=(const BitBoard& rhs) {
        lo &= rhs.lo;
        hi &= rhs.hi;
        return *this;
    }
    inline BitBoard& operator|=(const BitBoard& rhs) {
        lo |= rhs.lo;
        hi |= rhs.hi;
        return *this;
    }
    inline BitBoard operator&(const BitBoard& rhs) const {
        auto res = *this;
        res &= rhs;
        return res;
    }
    inline BitBoard operator|(const BitBoard& rhs) const {
        auto res = *this;
        res |= rhs;
        return res;
    }
    bool operator[](const int& idx) const {  // 読み取り専用
        return (bool)((idx < 55 ? lo >> idx : hi >> idx - 55) & 1);
    }
};

/*
namespace test {
    void CheckLinear() {
        auto linear = Linear<3, 4>();
        iota(linear.parameters.weight.Ravel().begin(), linear.parameters.bias.end(), 0.0f);
        auto input = array<float, 3>{3.0, -2.0, 1.0};
        auto output = array<float, 4>();
        linear.Forward(input, output);
        for (int i = 0; i < 4; i++) cout << output[i] << " \n"[i == 3];  // => 12, 19, 26, 33
    }
    void CheckTorusConv2d() {
        constexpr auto input_dim = 4;
        constexpr auto output_dim = 2;
        auto conv = TorusConv2d<input_dim, output_dim>();
        conv.parameters.weight[1].Fill(1.0);
        conv.parameters.weight[0][0][0][0] = 1.0f;
        conv.parameters.weight[0][0][0][1] = 2.0f;
        conv.parameters.weight[0][1][0][0] = 7.0f;
        conv.parameters.weight[1][0][2][1] = 11.0f;
        conv.parameters.bias[1] = -1.0f;
        conv.batchnorm.parameters.running_mean[0] = 3.0f;
        conv.batchnorm.parameters.running_var[0] = 0.25f;
        conv.batchnorm.parameters.weight[0] = -1.0f;
        conv.batchnorm.parameters.bias[0] = -5.0f;
        auto input = Tensor3<float, input_dim, 4, 5>();
        auto output = Tensor3<float, output_dim, 4, 5>();
        conv.parameters.weight.Print();
        input[0][0][0] = 1.0f;
        input[0][1][2] = 5.0f;
        input[1][2][3] = 3.0f;
        conv.Forward(input, output);
        cout << "Input:" << endl;
        input.Print();
        cout << "Output:" << endl;
        output.Print();
        // [[[  0.9999,   0.9999,   0.9999,   0.9999,   0.9999],
        //   [ -3.0000,  -1.0001,   0.9999,   0.9999,   0.9999],
        //   [  0.9999,   0.9999, -18.9997,  -8.9999,   0.9999],
        //   [  0.9999,   0.9999,   0.9999,   0.9999, -40.9993]],

        //  [[  0.0000,   5.0000,  53.9997,   4.0000,   0.0000],
        //   [  0.0000,   5.0000,   7.0000,   7.0000,   3.0000],
        //   [ -1.0000,   4.0000,   7.0000,   7.0000,   2.0000],
        //   [ 10.0000,   0.0000,   2.0000,   2.0000,   3.0000]]]
    }
    void CheckModel() {
        // TODO
    }
    // 100 回予測する時間の計測
    void CheckPredictionTime() {
        // TODO
        using namespace nagiss_library;
        auto model = GeeseNet<>();
        auto input = Tensor3<float, 17, 7, 11>();
        auto output_policy = array<float, 4>();
        auto output_value = 0.0f;
        auto rng = Random(time() * 1e9);
        auto t0 = time();
        // ランダムにパラメータをセットする
        // モデルがパラメータ以外の変数を持っていた場合使えない、あと sqrt 部分もやばい
        //for(auto ptr = (float*)&model; ptr != (float*)&model + sizeof(model) / sizeof(float); ptr++){
        //    *ptr = rng.random() * 1e-3;
        //}
        // この include がコンパイル時に激重なので必要なければコメントアウトする
        //#include "parameters.hpp"
        for (int i = 0; i < 100; i++) {
            for (auto&& v : input.Ravel()) v = rng.random();  // ランダムに入力をセットする
            input[0][rng.randint(7)][rng.randint(11)] = 1.0f;
            model.Forward(input, output_policy, output_value);
            for (int j = 0; j < 4; j++) cout << output_policy[j] << " \n"[j == 3];
            cout << output_value << "\n";
        }
        cout << "time=" << time() - t0 << endl;
    }
};
*/

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
    array<int, N_FEATURES> data;
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
    array<int, N_FEATURES> data;
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
    array<int, N_FEATURES+1> data;
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
    cout << "BOUNDARY = [";
    for (auto i = 0; i < N_FEATURES; i++) {
        cout << idx << ",";
        idx += MAX.data[i] - MIN.data[i] + 1;
    }
    cout << idx << "]" << endl;

    cout << "OFFSET = [";
    for (auto i = 0; i < N_FEATURES; i++) {
        cout << OFFSET.data[i] << ",";
    }
    cout << "]" << endl;
}

constexpr auto MAX_FEATURE_REACHABLE_CALCULATION = 8;

template<class IntStack1, class IntStack2>  // std::vector とだいたい同等の操作ができるクラス
void ExtractFeatures(
    const array<IntStack1, 4>& geese,
    const array<int, 2>& foods,
    const int& current_step,
    array<IntStack2, 4>& agent_features,    // output
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
    auto sorted_lengths = array<int, 4>();
    for (int i = 0; i < 4; i++) sorted_lengths[i] = geese[i].size();
    sort(sorted_lengths.begin(), sorted_lengths.end(), greater<>());

    // 前処理: future ステップ以内に到達可能な場所 (他 geese の頭が動かないと仮定)  // 正しくない場合もある (長さが 1 のときなど)
    auto not_occupied = BitBoard(occupied_bitboard);
    not_occupied.Invert();
    auto reachable_positions = array<array<BitBoard, MAX_FEATURE_REACHABLE_CALCULATION + 1>, 4>();  // 各 goose の 1 ~ 8 ステップ後に到達可能な場所
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
}  // namespace feature

struct Evaluator {
    Model<feature::NN_INPUT_DIM + 1> model;  // Python 側の都合でひとつ多く持つ
    Evaluator() {
        // モデルのパラメータ設定
        model.embed.parameters.weight.Ravel() = 
        
        // TODO
    }
    template<class T, int max_size>
    using Stack = nagiss_library::Stack<T, max_size>;
    auto evaluate(const array<Stack<int, 77>, 4>& geese, const array<int, 2>& foods, const int& current_step) {
        // モデルへの入出力用変数
        static auto agent_features = array<Stack<int, 100>, 4>();
        static auto condition_features = Stack<int, 100>();
        static auto output = Matrix<float, 4, 5>();

        feature::ExtractFeatures(geese, foods, current_step, agent_features, condition_features);
        model.Forward(agent_features, condition_features, output);
        return output;  // TODO: p とか v とかに分ける
    }
};

}// namespace evaluation_function


namespace tree_search {
using namespace std;
struct Duct {
    constexpr static int BIG = 100000;

    struct Node {
        State state;
        array<array<float, 4>, 4> policy;  // 各エージェントの方策
        array<array<float, 4>, 4> worth;   // 各エージェント視点の各手の価値
        int n;                             // このノードで行動を起こした回数 (= 到達回数 - 1)
        int n_children;                    // 子ノードの数
        int children_offset;               // 子ノード
        int node_type;                     // エージェントのノード (0) か、食べ物のノード (1) か

        Node(){}

        Node(const State& arg_state, Stack<Node*, BIG>& children_buffer) : state(arg_state), policy(), worth(), n(0), children_offset(), node_type(0) {
            policy[0][0] = -100.0;

            n_children = 子ノードの数を数える処理();
            children_offset = children_buffer.size();
            children_buffer.resize(children_buffer.size() + n_children);

            if (state.foods[0] == -1 || state.foods[1] == -1) node_type = 1;
        }

        bool Expanded() const {  // 既にモデルによって評価されているかを返す
            return policy[0][0] != -100.0;
        }

        int ChooseMove() {  // 例の式を使って (食べ物のノードの場合はランダムに) 手を選ぶ  // 手の全パターンをひとつの値で表す。全員が 3 方向に移動できるなら 0 から 80 までの値をとる。Move メソッドで具体的な手に変換できる

        }

        int Move(const int& idx_move, const int& idx_agent) {  // idx_move に対応する idx_agent 番目のエージェントの手を返す
            ASSERT_RANGE(idx_move, 0, n_children);
        }

        Node& KthChildren(Stack<Node, BIG>& node_buffer, Stack<Node*, BIG>& children_buffer, const int& k) {  // k 番目の行動によって遷移する子ノードを返す  // その子ノードが初めて遷移するものだった場合、新たに領域を確保してノードを作る
            ASSERT_RANGE(k, 0, n_children);
            Node* child = children_buffer[children_offset + k];
            if (child == nullptr) {
                // 領域を確保
                node_buffer.emplace(state.NextState(k));
                child = children_buffer[children_offset + k] = &node_buffer.back();
            }
            return *child;
        }

    };


    Stack<Node, BIG> node_buffer;  // スタック領域に持ちたくない…
    Stack<Node*, BIG> children_buffer;
    array<int, BIG> move_buffer;
    Model model;

    Duct(const State& arg_state) {
        node_buffer[0] = Node(arg_state);
        for (auto&& c : children_buffer) c = nullptr;
    }

    void Search() {
        while (true) {
            Iterate();
        }
    }

    Node& RootNode() {
        return node_buffer[0];
    }

    void Iterate() {
        // 根から葉に移動
        Node* v = &RootNode();
        Stack<int, 100> path;
        while (v->Expanded() || v->node_type == 1) {  // 展開されていない、エージェントのノードに到達したら抜ける
            int move_idx = v->ChooseMove();
            path.push(move_idx);
            v = &v->KthChildren(node_buffer, children_buffer, move_idx);
            if (v->state.Finished()) {  // 終局状態
                break;
            }
        }

        // 葉ノードの処理
        array<float, 4> value;
        if (v->state.Finished()) {
            // 決着がついた場合、順位に応じて value を設定
            value = v->state.Scores();
        }
        else {
            // 未探索のノードに到達した場合、評価関数を呼ぶ
            Node* leaf = v;
            const auto policy_value = model.Predict(leaf->state);  // struct{ array<array<float, 4>, 4> policy; array<float, 4> value; }
            const array<array<float, 4>, 4>& policy = policy_value.policy;
            value = policy_value.value;
            leaf->policy = policy_value.policy;
        }

        // 葉での評価結果を経路のノードに反映
        v = &RootNode();
        for (const auto& move_idx : path) {
            for (int idx_agent = 0; idx_agent < 4; idx_agent++) {
                v->worth[idx_agent][v->Move(move_idx, idx_agent)] += value[idx_agent];
                v->n++;
            }
            v = &v->KthChildren(node_buffer, children_buffer, move_idx);
        }

    }

};
}

GreedyAgent::GreedyAgent() {}

AgentResult GreedyAgent::run(const Stage& aStage, int aIndex) {
	AgentResult result;
	std::array<int, 4> can_actions;
	for (int i = 0; i < 4; ++i) {
		can_actions[i] = 1;
	}
	{ // 前ターンの反対の行動を取らない
		auto act = aStage.mLastActions;
		if (act[aIndex] == Action::NORTH) {
			can_actions[2] = 0;
		}
		else if (act[aIndex] == Action::EAST) {
			can_actions[3] = 0;
		}
		else if (act[aIndex] == Action::SOUTH) {
			can_actions[0] = 0;
		}
		else if (act[aIndex] == Action::WEST) {
			can_actions[1] = 0;
		}
	}
	// 現在身体のある位置に移動しない
	for (int i = 0; i < 4; ++i) {
		auto Pos = Translate(aStage.geese()[aIndex].items()[0], i);
		for (int j = 0; j < 4; ++j) {
			if (!aStage.geese()[j].isSurvive()) {
				continue;
			}
			for (auto aPos:aStage.geese()[j].items()) {
				if (Pos == aPos) {
					can_actions[i] = 0;
				}
			}
		}
	}
	// 相手の頭の隣接4マスには移動しない
	for (int i = 0; i < 4; ++i) {
		auto Pos = Translate(aStage.geese()[aIndex].items()[0], i);
		for (int j = 0; j < 4; ++j) {
			if (aIndex == j) {
				continue;
			}
			if (!aStage.geese()[j].isSurvive()) {
				continue;
			}
			for (int k = 0; k < 4; ++k) {
				auto aPos = Translate(aStage.geese()[j].items()[0], k);
				if (Pos == aPos) {
					can_actions[i] = 0;
				}
			}
		}
	}
	int opt_action = 0;
	for (int i = 0; i < 4; ++i) {
		if (can_actions[i]) {
			opt_action = i;
			break;
		}
	}
	// 食べ物に一番近い位置に移動する
	int min_food_distance = INF;
	for (int i = 0; i < 4; ++i) {
		if (can_actions[i]) {
			for (int j = 0; j < 2; ++j) {
				int result = min_Distance(aStage.foods()[j].pos(), Translate(aStage.geese()[aIndex].items()[0], i));
				if (min_food_distance > result) {
					min_food_distance = result;
					opt_action = i;
				}
			}
		}
	}
	result.mAction = opt_action;


    // 特徴量計算
    // 無理やり合わせてる　うーんこの
    {
        // 入力
        auto geese = std::array<Stack<int, 77>, 4>();
        for (int i = 0; i < 4; i++) {
            if (!aStage.geese()[i].isSurvive()) continue;
            for (const auto& p : aStage.geese()[i].items()) {
                geese[i].push(p.id);
            }
        }
        const auto foods = std::array<int, 2>{ aStage.foods()[0].pos().id, aStage.foods()[1].pos().id };
        const auto& current_step = aStage.mTurn;

        // 出力
        static auto agent_features = std::array<Stack<int, 100>, 4>();
        static auto condition_features = Stack<int, 100>();

        evaluation_function::feature::ExtractFeatures(geese, foods, current_step, agent_features, condition_features);
        result.mAgentFeatures = agent_features;
        result.mConditionFeatures = condition_features;
    }
	return result;
}

int GreedyAgent::min_Distance(Point aPos, Point bPos) {
	int result = 0;
	int row = std::abs(aPos.x - bPos.x);
	result += std::min(row, Parameter::rows - row);
	int column = std::abs(aPos.y - bPos.y);
	result += std::min(column, Parameter::columns - column);
	return result;
}

Point GreedyAgent::Translate(Point aPos, int Direction) {
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