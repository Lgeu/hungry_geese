#include "library.hpp"

namespace evaluation_function {

using namespace std;

template<class T, int dim1, int dim2>
struct Matrix{
    array<array<T, dim2>, dim1> data;
    array<T, dim2>& operator[](const int& idx){
        return data[idx];
    }
    const array<T, dim2>& operator[](const int& idx) const {
        return data[idx];
    }
    array<T, dim1 * dim2>& Ravel(){
        union U{
            array<array<T, dim2>, dim1> data;
            array<T, dim1 * dim2> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const array<T, dim1 * dim2>& Ravel() const {
        union U{
            array<array<T, dim2>, dim1> data;
            array<T, dim1 * dim2> raveled;
        };
        return ((U*)&data)->raveled;
    }
    void Fill(const T& fill_value) {
        fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    inline auto& operator+=(const Matrix& rhs) {
        for(int i=0; i<dim1*dim2; i++) Ravel()[i] += rhs.Ravel()[i];
		return *this;
	}
    void Print() const {
        for(int i=0; i<dim1; i++){
            for(int j=0; j<dim2; j++){
                cout << data[i][j] << " \n"[j==dim2-1];
            }
        }
    }
};

template<class T, int dim1, int dim2, int dim3>
struct Tensor3{
    array<Matrix<T, dim2, dim3>, dim1> data;
    Matrix<T, dim2, dim3>& operator[](const int& idx){
        return data[idx];
    }
    const Matrix<T, dim2, dim3>& operator[](const int& idx) const {
        return data[idx];
    }
    array<T, dim1 * dim2 * dim3>& Ravel(){
        union U{
            array<Matrix<T, dim2, dim3>, dim1> data;
            array<T, dim1 * dim2 * dim3> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const array<T, dim1 * dim2 * dim3>& Ravel() const {
        union U{
            array<Matrix<T, dim2, dim3>, dim1> data;
            array<T, dim1 * dim2 * dim3> raveled;
        };
        return ((U*)&data)->raveled;
    }
    template<int new_dim_1, int new_dim_2>
    Matrix<new_dim_1, new_dim_2>  View(){
        static_assert(dim1 * dim2 * dim3 == new_dim_1 * new_dim_2, "View の次元がおかしいよ");
        union U{
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
        for(int i1=0; i1<dim1; i1++)
            for(int i2=0; i2<dim2; i2++)
                for(int i3=0; i3<dim3; i3++)
                    permuted
                        [array<int, 3>{i1,i2,i3}[p0]]
                        [array<int, 3>{i1,i2,i3}[p1]]
                        [array<int, 3>{i1,i2,i3}[p2]] = data[i1][i2][i3];
        return permuted;
    }
    inline auto& operator+=(const Tensor3& rhs) {
        for(int i=0; i<dim1*dim2*dim3; i++) Ravel()[i] += rhs.Ravel()[i];
		return *this;
	}
    void Print() const {
        for(int i=0; i<dim1; i++){
            data[i].Print();
            if(i != dim1-1) cout << endl;
        }
    }
};

template<class T, int dim1, int dim2, int dim3, int dim4>
struct Tensor4{
    array<Tensor3<T, dim2, dim3, dim4>, dim1> data;

    Tensor3<T, dim2, dim3, dim4>& operator[](const int& idx){
        return data[idx];
    }
    const Tensor3<T, dim2, dim3, dim4>& operator[](const int& idx) const {
        return data[idx];
    }
    array<T, dim1 * dim2 * dim3 * dim4>& Ravel(){
        union U{
            array<Tensor3<T, dim2, dim3, dim4>, dim1> data;
            array<T, dim1 * dim2 * dim3 * dim4> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const array<T, dim1 * dim2 * dim3 * dim4>& Ravel() const {
        union U{
            array<Tensor3<T, dim2, dim3, dim4>, dim1> data;
            array<T, dim1 * dim2 * dim3 * dim4> raveled;
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
        for(int i1=0; i1<dim1; i1++)
            for(int i2=0; i2<dim2; i2++)
                for(int i3=0; i3<dim3; i3++)
                    for(int i4=0; i4<dim4; i4++)
                        permuted
                            [array<int, 4>{i1,i2,i3,i4}[p0]]
                            [array<int, 4>{i1,i2,i3,i4}[p1]]
                            [array<int, 4>{i1,i2,i3,i4}[p2]]
                            [array<int, 4>{i1,i2,i3,i4}[p3]] = data[i1][i2][i3][i4];
        return permuted;
    }
    inline auto& operator+=(const Tensor4& rhs) {
        for(int i=0; i<dim1*dim2*dim3*dim4; i++) Ravel()[i] += rhs.Ravel()[i];
		return *this;
	}
    void Print() const {
        for(int i=0; i<dim1; i++){
            data[i].Print();
            if(i != dim1-1) cout << endl << endl;
        }
    }
};

namespace F{
    template<class Tensor>
    void Relu_(Tensor& input){
        for(auto&& value : input.Ravel()){
            value = max(value, (typename remove_reference<decltype(input.Ravel()[0])>::type)0);
        }
    }
    template<typename T, unsigned siz>
    void Relu_(array<T, siz>& input){
        for(auto&& value : input){
            value = max(value, (T)0);
        }
    }
    template<class T, size_t siz>
    void Softmax_(array<T, siz>& input){
        auto ma = numeric_limits<float>::min();
        for(const auto& v : input) if(ma < v) ma = v;
        auto s = 0.0f;
        for(const auto& v : input) s += expf(v - ma);
        auto c = ma + logf(s);
        for(auto&& v : input) v = expf(v - c);
    }
}

template<int n_features>
struct BatchNorm2d{
    struct Parameter{
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
        for(int channel=0; channel<n_features; channel++){
            const auto coef = parameters.weight[channel] / sqrtf(parameters.running_var[channel] + 1e-5f);
            const auto bias_ = parameters.bias[channel] - coef * parameters.running_mean[channel];
            for(int y=0; y<height; y++){
                for(int x=0; x<width; x++){
                    input[channel][y][x] = input[channel][y][x] * coef + bias_;
                }
            }
        }
    }
};

template<int input_dim, int output_dim, int kernel_size=3>
struct TorusConv2d{
    struct Parameter{
        Tensor4<float, output_dim, input_dim, kernel_size, kernel_size> weight;
        array<float, output_dim> bias;
    } parameters;
    
    BatchNorm2d<output_dim> batchnorm;


    // コンストラクタ
    TorusConv2d() : parameters(), batchnorm() {}
    
    constexpr void SetParameters(){
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
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                for(int out_channel=0; out_channel<output_dim; out_channel++){
                    output[out_channel][y][x] = parameters.bias[out_channel];
                }
                for(int ky=0; ky<kernel_size; ky++){
                    auto y_from = y + ky - pad;
                    if (y_from < 0) y_from += height;
                    else if (y_from >= height) y_from -= height;
                    for(int kx=0; kx<kernel_size; kx++){
                        auto x_from = x + kx - pad;
                        if(x_from < 0) x_from += width;
                        else if (x_from >= width) x_from -= width;
                        for(int out_channel=0; out_channel<output_dim; out_channel++){
                            for(int in_channel=0; in_channel<input_dim; in_channel++){
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
struct Linear{
    struct Parameters{
        Matrix<float, out_features, in_features> weight;
        array<float, out_features> bias;
    } parameters;
    
    // コンストラクタ
    Linear() : parameters() {}

    void Forward(const array<float, in_features>& input, array<float, out_features>& output) const {
        output = parameters.bias;
        for(int out_channel=0; out_channel<out_features; out_channel++){
            for(int in_channel=0; in_channel<in_features; in_channel++){
                output[out_channel] += input[in_channel] * parameters.weight[out_channel][in_channel];
            }
        }
    }
};

template<int num_embeddings, int embedding_dim>
struct EmbeddingBag{
    // mode="sum" のみ対応
    struct Parameters{
        Matrix<float, num_embeddings, embedding_dim> weight;
    } parameters;

    // コンストラクタ
    EmbeddingBag() : parameters() {}

    template<class Vector>  // Stack<int, n> とか
    void Forward(const Vector& input, array<float, embedding_dim>& output) const {
        fill(output.begin(), output.end(), 0.0f);
        for(const auto& idx : input){
            for(int dim=0; dim<embedding_dim; dim++){
                output[dim] += parameters.weight[idx][dim];
            }
        }
    }
};

template<int n_agent_features, int n_condition_features, int out_dim=5, int hidden_1=256, int hidden_2=32>
struct Model{
    EmbeddingBag<n_agent_features, hidden_1> embed;
    EmbeddingBag<n_condition_features, hidden_1> embed_condition;
    Linear<hidden_1, hidden_2> linear_condition;
    Linear<hidden_1, hidden_2> linear_2;
    Linear<hidden_2, hidden_2> linear_3;
    Linear<hidden_1, out_dim> linear_4;
    
    // コンストラクタ
    Model() : embed(), embed_condition(), linear_condition(), linear_2(), linear_3(), linear_4() {}

    template<class Vector1, class Vector2>  // Stack<int, n> とか
    void Forward(const array<Vector, 4>& agent_features, const Vector2& condition_features, Matrix<float, 4, out_dim>& output){
        static auto agent_embedded = Matrix<float, 4, hidden_1>();
        for(int idx_agents=0; idx_agents<4; idx_agents++){
            embed.Forward(agent_features[idx_agents], agent_embedded[idx_agents]);
        }
        F::Relu_(agent_embedded);

        static auto condition_embedded = Matrix<float, hidden_1>();
        embed_condition(condition_features, condition_embedded);
        F::Relu_(condition_embedded);
        for(int idx_agents=0; idx_agents<4; idx_agents++){
            for(int dim=0; dim<hidden_1; dim++){
                condition_embedded[dim] += embed[idx_agents][dim];  // Vector 構造体を作るべきだった感
            }
        }
        static auto condition_hidden = array<float, hidden_2>();
        linear_condition(condition_embedded, condition_hidden);
        F::Relu_(condition_hidden);

        static auto hidden_state_2 = Matrix<float, 4, hidden_2>();
        for(int idx_agents=0; idx_agents<4; idx_agents++){
            linear_2(agent_embedded[idx_agents], hidden_state_2[idx_agents]);
        }
        for(int idx_agents=0; idx_agents<4; idx_agents++){
            for(int dim=0; dim<hidden_2; dim++){
                hidden_state_2[idx_agents][dim] += condition_hidden[dim];
            }
        }
        F::Relu_(hidden_state_2);

        static auto hidden_state_3 = Matrix<float, 4, hidden_2>();
        for(int idx_agents=0; idx_agents<4; idx_agents++){
            linear_3(hidden_state_2[idx_agents], hidden_state_3[idx_agents]);
        }
        F::Relu_(hidden_state_3);
        
        for(int idx_agents=0; idx_agents<4; idx_agents++){
            linear_4(hidden_state_3[idx_agents], output[idx_agents]);
        }
    }

};


struct BitBoard{
    using ull = unsigned long long;
    ull lo, hi;
    static constexpr ull mask_right_lo = 0b10000000000'10000000000'10000000000'10000000000'10000000000ull, mask_right_hi = 0b10000000000'10000000000ull;
    static constexpr ull mask_left_lo  = 0b00000000001'00000000001'00000000001'00000000001'00000000001ull, mask_left_hi  = 0b00000000001'00000000001ull;
    static constexpr ull mask_down_lo  = 0b11111111111'00000000000'00000000000'00000000000'00000000000ull, mask_down_hi  = 0b11111111111'00000000000ull;
    static constexpr ull mask_up_lo    = 0b00000000000'00000000000'00000000000'00000000000'11111111111ull, mask_up_hi    = 0b00000000000'11111111111ull;
    static constexpr ull mask_all_lo   = 0b11111111111'11111111111'11111111111'11111111111'11111111111ull, mask_all_hi   = 0b11111111111'11111111111ull;
    inline BitBoard() : lo(), hi() {}
    inline BitBoard(const int& idx) : lo(), hi() { Flip(idx); }
    inline void Print() const {
        for(int y=0; y<5; y++){
            for(int x=0; x<11; x++){
                cout << ((lo>>y*11+x)&1);
            }
            cout << endl;
        }
        for(int y=0; y<2; y++){
            for(int x=0; x<11; x++){
                cout << ((hi>>y*11+x+55)&1);
            }
            cout << endl;
        }
    }
    inline void Flip(const int& idx){
        if(idx<55) lo ^= 1ull << idx;
        else hi ^= 1ull << idx-55;
    }
    inline void ShiftRight(){
        const auto masked_lo = lo & mask_right_lo;  // 右端を取り出す
        const auto masked_hi = hi & mask_right_hi;
        lo ^= masked_lo;                            // 右端を消す
        hi ^= masked_hi;
        lo <<= 1;                                   // 右にシフト
        hi <<= 1;
        lo |= masked_lo >> 10;                     // 右端にあったものを左端に持ってくる
        hi |= masked_hi >> 10;
    }
    inline void ShiftLeft(){
        const auto masked_lo = lo & mask_left_lo;
        const auto masked_hi = hi & mask_left_hi;
        lo ^= masked_lo;
        hi ^= masked_hi;
        lo >>= 1;
        hi >>= 1;
        lo |= masked_lo << 10;
        hi |= masked_hi << 10;
    }
    inline void ShiftDown(){
        const auto masked_lo = lo & mask_down_lo;
        const auto masked_hi = hi & mask_down_hi;
        lo ^= masked_lo;
        hi ^= masked_hi;
        lo <<= 11;
        hi <<= 11;
        lo |= masked_hi >> 11;
        hi |= masked_lo >> 44;
    }
    inline void ShiftUp(){
        const auto masked_lo = lo & mask_up_lo;
        const auto masked_hi = hi & mask_up_hi;
        lo >>= 11;
        hi >>= 11;
        lo |= masked_hi << 44;
        hi |= masked_lo << 11;
    }
    inline void Invert(){
        lo ^= mask_all_lo;
        hi ^= mask_all_hi;
    }
    inline int Popcount() const {
        using nagiss_library::popcount;
        return popcount(lo) + popcount(hi);
    }
    inline int Neighbor12(const int& idx) const {
        // 00100
        // 01110
        // 11011
        // 01110
        // 00100
        // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        int res = 0;
        constexpr auto dyxs = array<Vec2<int>, 12>{Vec2<int>{-2,0},{-1,-1},{-1,0},{-1,1},{0,-2},{0,-1},{0,1},{0,2},{1,-1},{1,0},{1,1},{2,0}};
        auto yx0 = Vec2<int>{idx / 11, idx % 11};
        for(int i=0; i<12; i++){
            auto yx = yx0 + dyxs[i];
            if (yx.y < 0) yx.y += 7;
            else if(yx.y >= 7) yx.y -= 7;
            if (yx.x < 0) yx.x += 11;
            else if(yx.x >= 11) yx.x -= 11;
            auto idx_ = yx.y * 11 + yx.x;
            res |= ((idx_ < 55 ? lo >> idx_ : hi >> idx_-55) & 1) << i;
        }
        return res;
    }
    inline bool Empty() const {
        return lo == 0ull && hi == 0ull;
    }
    inline BitBoard& operator&=(const BitBoard& rhs){
        lo &= rhs.lo;
        hi &= rhs.hi;
        return *this;
    }
    inline BitBoard& operator|=(const BitBoard& rhs){
        lo |= rhs.lo;
        hi &= rhs.hi;
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
        return (bool)((idx < 55 ? lo >> idx : hi >> idx-55) & 1);
    }
};


namespace test{
    void CheckLinear(){
        auto linear = Linear<3, 4>();
        iota(linear.parameters.weight.Ravel().begin(), linear.parameters.bias.end(), 0.0f);
        auto input = array<float, 3>{3.0, -2.0, 1.0};
        auto output = array<float, 4>();
        linear.Forward(input, output);
        for(int i=0; i<4; i++) cout << output[i] << " \n"[i==3];  // => 12, 19, 26, 33
    }
    void CheckTorusConv2d(){
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
    void CheckModel(){
        // TODO
    }
    // 100 回予測する時間の計測
    void CheckPredictionTime(){
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
        for(int i=0; i<100; i++){
            for(auto&& v : input.Ravel()) v = rng.random();  // ランダムに入力をセットする
            input[0][rng.randint(7)][rng.randint(11)] = 1.0f;
            model.Forward(input, output_policy, output_value);
            for(int j=0; j<4; j++) cout << output_policy[j] << " \n"[j==3];
            cout << output_value << "\n";
        }
        cout << "time=" << time() - t0 << endl;
    }
};


namespace feature{
namespace offset{
    auto HOGE = 123;
}

template<class Stack>  // std::vector とだいたい同等の操作ができるクラス
void ExtractFeatures(
    const array<Stack<int, 77>, 4>& geese,
    const array<int, 2>& foods,
    const BitBoard& occupied_bitboard,  // いずれかの geese がいる位置が 1、それ以外が 0
    Info info,  // TODO ターン数など
    array<Stack, 4>& agent_features,    // output
    Stack& condition_features           // output
){
    using nagiss_library::Vec2;
    using nagiss_library::clipped;
    // この周辺を効率化する方法はいろいろ考えられるけど、加算回数が減るだけなのでどれくらいいい感じになるのか不明
    // 差分計算（親ノード、類似ノード）
    // ベクトル量子化

    // 初期化
    for(int i=0; i<4; i++) agent_features[i].clear();
    condition_features.clear();
    
    // 前処理
    auto sorted_lengths = array<int, 4>();
    for(int i=0; i<4; i++) sorted_lengths[i] = geese[i].size();
    sort(sorted_lengths.begin(), sorted_lengths.end(), greater<>);

    // future ステップ以内に到達可能な場所 (他 geese の頭が動かないと仮定)
    auto not_occupied = BitBoard(occupied_bitboard);
    not_occupied.Invert();
    constexpr auto MAX_FEATURE_REACHABLE_CALCULATION = 5;
    auto reachable_positions = array<array<BitBoard, MAX_FEATURE_REACHABLE_CALCULATION+1>, 4>()  // 各 goose の 1 ~ 5 ステップ後に到達可能な場所
    for(int i=0; i<4; i++){
        if (geese[i].size() == 0) continue;
        reachable_positions[i][0].Flip(geese[i].front());
    }
    for(int future=1, clearing_count=1; future<=MAX_FEATURE_REACHABLE_CALCULATION; future++, clearing_count++){
        // 短くなる処理
        for(int i=0; i<4; i++){
            auto geese_i_clearing_idx = geese[i].size() - clearing_count;
            if(geese_i_clearing_idx < 0) continue;
            const auto& idx = geese[i][geese_i_clearing_idx]
            not_occupied.Flip(idx);
            ASSERT(not_occupied[idx], "しっぽを消そうと思ったらしっぽが無かったよ");
        }
        // もう一回短くなる
        if((info.current_step + future) % 40 == 0){  // この条件合ってる？要確認
            clearing_count++;
            for(int i=0; i<4; i++){
                auto geese_i_clearing_idx = geese[i].size() - clearing_count;
                if(geese_i_clearing_idx < 0) continue;
                const auto& idx = geese[i][geese_i_clearing_idx]
                not_occupied.Flip(idx);
                ASSERT(not_occupied[idx], "しっぽを消そうと思ったらしっぽが無かったよ");
            }
        }
        for(int i=0; i<4; i++){
            if (geese[i].size() == 0) continue;
            const auto& prev_reachable_positions = reachable_positions[i][future-1];
            auto&       next_reachable_positions = reachable_positions[i][future];
            next_reachable_positions = prev_reachable_positions;
            auto tmp = prev_reachable_positions;  tmp.ShiftRight();  next_reachable_positions |= tmp;
                 tmp = prev_reachable_positions;  tmp.ShiftLeft();   next_reachable_positions |= tmp;
                 tmp = prev_reachable_positions;  tmp.ShiftDown();   next_reachable_positions |= tmp;
                 tmp = prev_reachable_positions;  tmp.ShiftUp();     next_reachable_positions |= tmp;
            next_reachable_positions &= not_occupied;
        }
    }

    // 各エージェント視点の特徴量
    for(int idx_agents=0; idx_agents<4; idx_agents++){
        const auto& goose = geese[idx_agnents];
        if(goose.size() == 0) continue;  // もう脱落していた場合、飛ばす

        // 基本的な情報を抽出
        const auto& head = goose.front();
        const auto& tail = goose.back();
        auto& features = agent_features[idx_agents];
        const auto head_vec = Vec2<int>(head/11, head%11);
        const auto tail_vec = Vec2<int>(tail/11, tail%11);

        // 距離 2 までの全パターン
        const auto pattern = occupied_bitboard.Neighbor12(head);
        ASSERT_RANGE(pattern, 0, 1<<12);
        features.push(offset::NEIGHBOR12 + pattern);

        // goose の長さ
        const auto length = goose.size();
        ASSERT_RANGE(pattern, 1, 78);
        features.push(offset::LENGTH + length);

        // [1-4] 番目に長い goose との長さの差
        for(int rank=0; rank<4; rank++){
            auto difference_length = length - sorted_lengths[rank];
            difference_length = clipped(difference_length, -10, 10);
            features.push(offset::DIFFERENCE_LENGTH[rank] + difference_length);
        }
        
        // しっぽの位置
        auto CalcRelativePosition = [](const int& s, const int& t){
            // s から見た t の位置  // 遅かったら表引きに書き換える
            const auto s_vec = Vec2<int>(s/11, s%11);
            const auto t_vec = Vec2<int>(t/11, t%11);
            auto relative_positioin_vec = t_vec - s_vec;
            if (relative_positioin_vec.y < 0) relative_positioin_vec.y += 7;
            if (relative_positioin_vec.x < 0) relative_positioin_vec.x += 11;
            ASSERT_RANGE(relative_positioin_vec.y, 0, 7);
            ASSERT_RANGE(relative_positioin_vec.x, 0, 11);
            const auto relative_position = relative_positioin_vec.y * 11 + relative_positioin_vec.x;
            return relative_position;
        };
        const auto relative_tail = CalcRelativePosition(head, tail);
        ASSERT_RANGE(relative_tail, 0, 77);  // 長さが 1 の場合に頭と同じ (0) になる
        features.push(offset::RELATIVE_POSITION_TAIL + relative_tail);

        // 敵の頭の位置・しっぽから見た敵の頭の位置
        for(int opponent=0; opponent<4; opponent++){
            if (opponent == idx_agents || geese[opponent].size() == 0) continue;
            const auto& opponent_head = geese[opponent].front();
            const auto relative_opponent_head = CalcRelativePosition(head, opponent_head);
            ASSERT_RANGE(relative_opponent_head, 1, 77);
            features.push(offset::RELATIVE_POSITION_OPPONENT_HEAD + relative_opponent_head);

            const auto relative_opponent_head_from_tail =  CalcRelativePosition(tail, opponent_head);
            ASSERT_RANGE(relative_opponent_head_from_tail, 1, 77);
            features.push(offset::RELATIVE_POSITION_OPPONENT_HEAD_FROM_TAIL + relative_opponent_head_from_tail);
        }

        // 食べ物の位置
        for (int idx_foods=0; idx_foods<foods.size(); idx_foods++){
            const auto relative_food = CalcRelativePosition(head, foods[idx_foods]);
            ASSERT_RANGE(relative_food, 1, 77);
            features.push(offset::RELATIVE_POSITION_FOOD + relative_food);
        }

        // 行動履歴
        auto history = 0;
        auto direction = 0;
        for(int i=0; i<4; i++){
            if (goose.size() <= i+1) {
                direction ^= 2;  // goose が短い場合は前回と逆向きに入れる
            } else {
                const auto relative_position = CalcRelativePosition(goose[i+1], goose[i]);
                switch(relative_position){
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
                    ASSERT(false, "何か間違ってるよ")
                }
            }
            history |= direction << i * 2;
        }
        ASSERT_RANGE(history, 0, 4*4*4*4);
        features.push(offset::MOVE_HISTORY + history);

        // 平面上に置いたときのしっぽの相対位置  // 差分計算可能
        auto relative_tail_on_plane_x = 0;
        auto relative_tail_on_plane_y = 0;
        auto old_yx = Vec2<int>(head/11, head%11);
        for(int i=0; i<goose.size()-1; i++){  // 3 つずつずらした方が効率的だけど面倒！
            const auto relative_position = CalcRelativePosition(goose[i+1], goose[i]);
            switch(relative_position){
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
                ASSERT(false, "何か間違ってるよ")
            }
        }
        relative_tail_on_plane_x = clipped(relative_tail_on_plane_x, -30, 30);
        relative_tail_on_plane_y = clipped(relative_tail_on_plane_y, -30, 30);
        features.push(offset::RELATIVE_POSITION_TAIL_ON_PLANE_X + relative_tail_on_plane_x);
        features.push(offset::RELATIVE_POSITION_TAIL_ON_PLANE_Y + relative_tail_on_plane_y);

        // n ステップ以内に到達可能な場所の数 (他 geese の頭が動かないと仮定)
        for(int n=1; n<=MAX_FEATURE_REACHABLE_CALCULATION; n++){
            auto n_reachable_positions_within_n_steps = reachable_positions[idx_agents][n].Popcount();
            ASSERT_RANGE(n_reachable_positions_within_n_steps, 1, array<int, MAX_FEATURE_REACHABLE_CALCULATION>{5, 13, 25, 40, 57}[n-1]);  // これも合ってるか？？
            features.push(offset::N_REACHABLE_POSITIONS_WITHIN_N_STEPS[n-1] + n_reachable_positions_within_n_steps);
        }

        // n ステップ以内に到達可能な場所が被ってる敵の数
        for(int n=1; n<=MAX_FEATURE_REACHABLE_CALCULATION; n++){
            auto n_opponents_sharing_reachable_positions_within_n_steps = 0;
            for(int i=0; i<4; i++){
                if (i == idx_agents || geese[i].size() == 0) continue;
                n_opponents_sharing_reachable_positions_within_n_steps += (int)!(reachable_positions[idx_agents][n] & reachable_positions[i][n]).Empty();
            }
            ASSERT_RANGE(n_opponents_sharing_reachable_positions_within_n_steps, 0, 4);
            features.push(offset::N_OPPONENTS_SHARING_REACHABLE_POSITIONS_WITHIN_N_STEPS[n-1] + n_opponents_sharing_reachable_positions_within_n_steps);
        }

        // n ステップ以内に自分だけが到達可能な場所の数
        for(int n=1; n<=MAX_FEATURE_REACHABLE_CALCULATION; n++){
            auto not_opponents_reachable = BitBoard();
            for(int i=0; i<4; i++){
                if (i == idx_agents || geese[i].size() == 0) continue;
                not_opponents_reachable |= reachable_positions[i][n];
            }
            not_opponents_reachable.Invert();
            auto n_exclusively_reachable_positions_within_n_steps = (reachable_positions[idx_agents][n] & not_opponents_reachable).Popcount();
            ASSERT_RANGE(n_exclusively_reachable_positions_within_n_steps, 0, array<int, MAX_FEATURE_REACHABLE_CALCULATION>{5, 13, 25, 40, 57}[n-1]);  // 合ってる？
            features.push(offset::N_EXCLUSIVELY_REACHABLE_POSITIONS_WITHIN_N_STEPS[n-1] + n_exclusively_reachable_positions_within_n_steps);
        }




    }

    // 全体の特徴量
    // TODO

}
}  // namespace feature

struct Evaluator{
    Model<nnn, nnnn> model;
    Evaluator(){
        // パラメータ設定
        // TODO
    }
    template<class T, int max_size>
    using Stack = nagiss_library::Stack<T, max_size>;
    auto evaluate(const array<Stack<int, 77>, 4>& geese, const array<int, 2>& foods, const BitBoard& occupied_bitboard) const {
        // モデルへの入出力用変数
        static auto agent_features = array<Stack<int, 100>, 4>();
        static auto condition_features = Stack<int, 100>();
        static auto output = Matrix<float, 4, 5>()
        
        feature::ExtractFeatures(geese, foods, occupied_bitboard, agent_features, condition_features);
        model.Forward(agent_features, agent_features, output);
        return output;  // TODO: p とか v とかに分ける
    }
};

}  // namespace evaluation_function

int main(){
    namespace test = evaluation_function::test;
    //test::CheckLinear();
    //test::CheckTorusConv2d();
    test::CheckGeeseNet();
    //test::CheckGeeseNetTime();
}

// 12 近傍、pext 命令と長さ 2^12 の配列でなんとかなるっぽい
