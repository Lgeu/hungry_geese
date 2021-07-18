#ifndef NAGISS_LIBRARY_HPP
#define NAGISS_LIBRARY_HPP
#include<iostream>
#include<iomanip>
#include<vector>
#include<set>
#include<map>
#include<unordered_set>
#include<unordered_map>
#include<algorithm>
#include<numeric>
#include<limits>
#include<bitset>
#include<functional>
#include<type_traits>
#include<queue>
#include<stack>
#include<array>
#include<random>
#include<utility>
#include<cstdio>
#include<cstdlib>
#include<ctime>
#include<string>
#include<sstream>
#include<chrono>
#include<climits>
#ifdef _MSC_VER
#include<intrin0.h>
#endif

#ifdef __GNUC__
//#pragma GCC target("avx2")
//#pragma GCC target("sse4")
//#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx,avx,tune=native")
//#pragma GCC optimize("O3")
//#pragma GCC optimize("Ofast")
//#pragma GCC optimize("unroll-loops")
#endif

// ========================== macroes ==========================

#define rep(i,n) for(ll (i)=0; (i)<(n); (i)++)
#define rep1(i,n) for(ll (i)=1; (i)<=(n); (i)++)
#define rep3(i,s,n) for(ll (i)=(s); (i)<(n); (i)++)

//#define NDEBUG

#ifndef NDEBUG
#define ASSERT(expr, ...) \
        do { \
            if(!(expr)){ \
                printf("%s(%d): Assertion failed.\n", __FILE__, __LINE__); \
                printf(__VA_ARGS__); \
                abort(); \
            } \
        } while (false)
#else
#define ASSERT(...)
#endif

#define ASSERT_RANGE(value, left, right) \
    ASSERT((left <= value) && (value < right), \
        "`%s` (%d) is out of range [%d, %d)", #value, value, left, right)

#define CHECK(var) do{ std::cout << #var << '=' << var << endl; } while (false)

// ========================== utils ==========================

namespace nagiss_library{

using namespace std;
using ll = long long;
constexpr double PI = 3.1415926535897932;

template<class T, class S> inline bool chmin(T& m, S q) {
    if (m > q) { m = q; return true; }
    else return false;
}

template<class T, class S> inline bool chmax(T& m, const S q) {
    if (m < q) { m = q; return true; }
    else return false;
}

// クリッピング  // clamp (C++17) と等価
template<class T> inline T clipped(const T& v, const T& low, const T& high) {
    return min(max(v, low), high);
}

// 2 次元ベクトル
template<typename T> struct Vec2 {
    /*
    y 軸正は下方向
    x 軸正は右方向
    回転は時計回りが正（y 軸正を上と考えると反時計回りになる）
    */
    T y, x;
    constexpr inline Vec2() = default;
    constexpr Vec2(const T& arg_y, const T& arg_x) : y(arg_y), x(arg_x) {}
    inline Vec2(const Vec2&) = default;  // コピー
    inline Vec2(Vec2&&) = default;  // ムーブ
    inline Vec2& operator=(const Vec2&) = default;  // 代入
    inline Vec2& operator=(Vec2&&) = default;  // ムーブ代入
    template<typename S> constexpr inline Vec2(const Vec2<S>& v) : y((T)v.y), x((T)v.x) {}
    inline Vec2 operator+(const Vec2& rhs) const {
        return Vec2(y + rhs.y, x + rhs.x);
    }
    inline Vec2 operator+(const T& rhs) const {
        return Vec2(y + rhs, x + rhs);
    }
    inline Vec2 operator-(const Vec2& rhs) const {
        return Vec2(y - rhs.y, x - rhs.x);
    }
    template<typename S> inline Vec2 operator*(const S& rhs) const {
        return Vec2(y * rhs, x * rhs);
    }
    inline Vec2 operator*(const Vec2& rhs) const {  // x + yj とみなす
        return Vec2(x * rhs.y + y * rhs.x, x * rhs.x - y * rhs.y);
    }
    template<typename S> inline Vec2 operator/(const S& rhs) const {
        ASSERT(rhs != 0.0, "Zero division!");
        return Vec2(y / rhs, x / rhs);
    }
    inline Vec2 operator/(const Vec2& rhs) const {  // x + yj とみなす
        return (*this) * rhs.inv();
    }
    inline Vec2& operator+=(const Vec2& rhs) {
        y += rhs.y;
        x += rhs.x;
        return *this;
    }
    inline Vec2& operator-=(const Vec2& rhs) {
        y -= rhs.y;
        x -= rhs.x;
        return *this;
    }
    template<typename S> inline Vec2& operator*=(const S& rhs) const {
        y *= rhs;
        x *= rhs;
        return *this;
    }
    inline Vec2& operator*=(const Vec2& rhs) {
        *this = (*this) * rhs;
        return *this;
    }
    inline Vec2& operator/=(const Vec2& rhs) {
        *this = (*this) / rhs;
        return *this;
    }
    inline bool operator!=(const Vec2& rhs) const {
        return x != rhs.x || y != rhs.y;
    }
    inline bool operator==(const Vec2& rhs) const {
        return x == rhs.x && y == rhs.y;
    }
    inline void rotate(const double& rad) {
        *this = rotated(rad);
    }
    inline Vec2<double> rotated(const double& rad) const {
        return (*this) * rotation(rad);
    }
    static inline Vec2<double> rotation(const double& rad) {
        return Vec2(sin(rad), cos(rad));
    }
    static inline Vec2<double> rotation_deg(const double& deg) {
        return rotation(PI * deg / 180.0);
    }
    inline Vec2<double> rounded() const {
        return Vec2<double>(round(y), round(x));
    }
    inline Vec2<double> inv() const {  // x + yj とみなす
        const double norm_sq = l2_norm_square();
        ASSERT(norm_sq != 0.0, "Zero division!");
        return Vec2(-y / norm_sq, x / norm_sq);
    }
    inline double l2_norm() const {
        return sqrt(x * x + y * y);
    }
    inline double l2_norm_square() const {
        return x * x + y * y;
    }
    inline T l1_norm() const {
        return std::abs(x) + std::abs(y);
    }
    inline double abs() const {
        return l2_norm();
    }
    inline double phase() const {  // [-PI, PI) のはず
        return atan2(y, x);
    }
    inline double phase_deg() const {  // [-180, 180) のはず
        return phase() / PI * 180.0;
    }
};
template<typename T, typename S> inline Vec2<T> operator*(const S& lhs, const Vec2<T>& rhs) {
    return rhs * lhs;
}
template<typename T> ostream& operator<<(ostream& os, const Vec2<T>& vec) {
    os << vec.y << ' ' << vec.x;
    return os;
}

// 乱数
struct Random {
    using ull = unsigned long long;
    ull seed;
    inline Random(ull aSeed) : seed(aSeed) {
        ASSERT(seed != 0ull, "Seed should not be 0.");
    }
    const inline ull& next() {
        seed ^= seed << 9;
        seed ^= seed >> 7;
        return seed;
    }
    // (0.0, 1.0)
    inline double random() {
        return (double)next() / (double)ULLONG_MAX;
    }
    // [0, right)
    inline int randint(const int right) {
        return next() % (ull)right;
    }
    // [left, right)
    inline int randint(const int left, const int right) {
        return next() % (ull)(right - left) + left;
    }
};


// キュー
template<class T, int max_size> struct Queue {
    array<T, max_size> data;
    int left, right;
    inline Queue() : data(), left(0), right(0) {}
    inline Queue(initializer_list<T> init) :
        data(init.begin(), init.end()), left(0), right(init.size()) {}

    inline bool empty() const {
        return left == right;
    }
    inline void push(const T& value) {
        data[right] = value;
        right++;
    }
    inline void pop() {
        left++;
    }
    const inline T& front() const {
        return data[left];
    }
    template <class... Args> inline void emplace(const Args&... args) {
        data[right] = T(args...);
        right++;
    }
    inline void clear() {
        left = 0;
        right = 0;
    }
    inline int size() const {
        return right - left;
    }
};


// スタック
template<class T, int max_size> struct Stack {
    array<T, max_size> data;
    int right;
    inline Stack() : data(), right(0) {}
    inline Stack(const int n) : data(), right(0) { resize(n); }
    inline Stack(const int n, const T& val) : data(), right(0) { resize(n, val); }
    inline Stack(initializer_list<T> init) :
        data(init.begin(), init.end()), right(init.size()) {}
    inline Stack(const Stack& rhs) : data(), right(rhs.right) {  // コピー
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
    }
    Stack& operator=(const Stack& rhs) {
        right = rhs.right;
        for (int i = 0; i < right; i++) {
            data[i] = rhs.data[i];
        }
        return *this;
    }
    Stack& operator=(const vector<T>& rhs) {
        right = (int)rhs.size();
        ASSERT(right <= max_size, "too big vector");
        for (int i = 0; i < right; i++) {
            data[i] = rhs[i];
        }
        return *this;
    }
    Stack& operator=(Stack&&) = default;
    inline bool empty() const {
        return 0 == right;
    }
    inline void push(const T& value) {
        ASSERT_RANGE(right, 0, max_size);
        data[right] = value;
        right++;
    }
    inline T pop() {
        right--;
        ASSERT_RANGE(right, 0, max_size);
        return data[right];
    }
    const inline T& top() const {
        return data[right - 1];
    }
    template <class... Args> inline void emplace(const Args&... args) {
        ASSERT_RANGE(right, 0, max_size);
        data[right] = T(args...);
        right++;
    }
    inline void clear() {
        right = 0;
    }
    inline void resize(const int& sz) {
        ASSERT_RANGE(sz, 0, max_size + 1);
        for (; right < sz; right++) {
            data[right].~T();
            new(&data[right]) T();
        }
        right = sz;
    }
    inline void resize(const int& sz, const T& fill_value) {
        ASSERT_RANGE(sz, 0, max_size + 1);
        for (; right < sz; right++) {
            data[right].~T();
            new(&data[right]) T(fill_value);
        }
        right = sz;
    }
    inline int size() const {
        return right;
    }
    inline T& operator[](const int n) {
        ASSERT_RANGE(n, 0, right);
        return data[n];
    }
    inline const T& operator[](const int n) const {
        ASSERT_RANGE(n, 0, right);
        return data[n];
    }
    inline T* begin() {
        return (T*)data.data();
    }
    inline const T* begin() const {
        return (const T*)data.data();
    }
    inline T* end() {
        return (T*)data.data() + right;
    }
    inline const T* end() const {
        return (const T*)data.data() + right;
    }
    inline T& front() {
        ASSERT(right > 0, "no data.");
        return data[0];
    }
    const inline T& front() const {
        ASSERT(right > 0, "no data.");
        return data[0];
    }
    inline T& back() {
        ASSERT(right > 0, "no data.");
        return data[right - 1];
    }
    const inline T& back() const {
        ASSERT(right > 0, "no data.");
        return data[right - 1];
    }

    inline vector<T> ToVector() {
        return vector<T>(begin(), end());
    }
};


// 時間 (秒)
inline double time() {
    return static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now().time_since_epoch()).count()) * 1e-9;
}


// 重複除去
template<typename T> inline void deduplicate(vector<T>& vec) {
    sort(vec.begin(), vec.end());
    vec.erase(unique(vec.begin(), vec.end()), vec.end());
}


template<typename T> inline int search_sorted(const vector<T>& vec, const T& a) {
    return lower_bound(vec.begin(), vec.end(), a) - vec.begin();
}


// popcount  // SSE 4.2 を使うべき
inline int popcount(const unsigned int& x) {
#ifdef _MSC_VER
    return (int)__popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}
inline int popcount(const unsigned long long& x) {
#ifdef _MSC_VER
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}

// x >> n & 1 が 1 になる最小の n ( x==0 は未定義 )
inline int CountRightZero(const unsigned int& x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward(&r, x);
    return (int)r;
#else
    return __builtin_ctz(x);
#endif
}
inline int CountRightZero(const unsigned long long& x) {
#ifdef _MSC_VER
    unsigned long r;
    _BitScanForward64(&r, x);
    return (int)r;
#else
    return __builtin_ctzll(x);
#endif
}
}  // namespace nagiss_library

#endif  // NAGISS_LIBRARY_HPP

namespace hungry_geese {
// Parameter
static const int columns = 11;
static const int rows = 7;

// Point
struct Cpoint {
    // コンストラクタ
    Cpoint() : mC() {};
    Cpoint(int aX, int aY) {
        mC = aX * columns + aY;
    }
    Cpoint(int aId) {
        mC = aId;
    }
    // メンバ変数
    signed char mC;
    // 呼び出し
    int X() const {
        return (int)mC / columns;
    }
    int Y() const {
        return (int)mC % columns;
    }
    int Id() const {
        return (int)mC;
    }
    // 演算子オーバーロード
    Cpoint& operator= (const Cpoint &aPos) {
        mC = aPos.Id();
        return *this;
    }
    bool operator== (const Cpoint &aPos) const {
        return (mC == aPos.Id());
    }
};
}

namespace evaluation_function {

using namespace nagiss_library;

template<class T, int dim1, int dim2>
struct Matrix{
    std::array<std::array<T, dim2>, dim1> data;
    std::array<T, dim2>& operator[](const int& idx){
        return data[idx];
    }
    const std::array<T, dim2>& operator[](const int& idx) const {
        return data[idx];
    }
    std::array<T, dim1 * dim2>& Ravel(){
        union U{
            std::array<std::array<T, dim2>, dim1> data;
            std::array<T, dim1 * dim2> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const std::array<T, dim1 * dim2>& Ravel() const {
        union U{
            std::array<std::array<T, dim2>, dim1> data;
            std::array<T, dim1 * dim2> raveled;
        };
        return ((U*)&data)->raveled;
    }
    void Fill(const T& fill_value) {
        std::fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    inline auto& operator+=(const Matrix& rhs) {
        for(int i=0; i<dim1*dim2; i++) Ravel()[i] += rhs.Ravel()[i];
        return *this;
    }
    void Print() const {
        for(int i=0; i<dim1; i++){
            for(int j=0; j<dim2; j++){
                std::cout << data[i][j] << " \n"[j==dim2-1];
            }
        }
    }
};

template<class T, int dim1, int dim2, int dim3>
struct Tensor3{
    std::array<Matrix<T, dim2, dim3>, dim1> data;
    Matrix<T, dim2, dim3>& operator[](const int& idx){
        return data[idx];
    }
    const Matrix<T, dim2, dim3>& operator[](const int& idx) const {
        return data[idx];
    }
    std::array<T, dim1 * dim2 * dim3>& Ravel(){
        union U{
            std::array<Matrix<T, dim2, dim3>, dim1> data;
            std::array<T, dim1 * dim2 * dim3> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const std::array<T, dim1 * dim2 * dim3>& Ravel() const {
        union U{
            std::array<Matrix<T, dim2, dim3>, dim1> data;
            std::array<T, dim1 * dim2 * dim3> raveled;
        };
        return ((U*)&data)->raveled;
    }
    void Fill(const T& fill_value) {
        std::fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    template<int p0, int p1, int p2>
    auto Permute() const {
        // d0 が 1 の場合、もともと 1 次元目だったところが 0 次元目に来る
        constexpr static auto d = std::array<int, 3>{dim1, dim2, dim3};
        auto permuted = Tensor3<T, d[p0], d[p1], d[p2]>();
        for(int i1=0; i1<dim1; i1++)
            for(int i2=0; i2<dim2; i2++)
                for(int i3=0; i3<dim3; i3++)
                    permuted
                        [std::array<int, 3>{i1,i2,i3}[p0]]
                        [std::array<int, 3>{i1,i2,i3}[p1]]
                        [std::array<int, 3>{i1,i2,i3}[p2]] = data[i1][i2][i3];
        return permuted;
    }
    /*
    auto Permute120() const {
        auto permuted = Tensor3<T, dim2, dim3, dim1>();
        for(int i1=0; i1<dim1; i1++){
            for(int i2=0; i2<dim2; i2++){
                for(int i3=0; i3<dim3; i3++){
                    permuted[i2][i3][i1] = data[i1][i2][i3];
                }
            }
        }
        return permuted;
    }*/
    inline auto& operator+=(const Tensor3& rhs) {
        for(int i=0; i<dim1*dim2*dim3; i++) Ravel()[i] += rhs.Ravel()[i];
        return *this;
    }
    void Print() const {
        for(int i=0; i<dim1; i++){
            data[i].Print();
            if(i != dim1-1) std::cout << std::endl;
        }
    }
};

template<class T, int dim1, int dim2, int dim3, int dim4>
struct Tensor4{
    std::array<Tensor3<T, dim2, dim3, dim4>, dim1> data;

    Tensor3<T, dim2, dim3, dim4>& operator[](const int& idx){
        return data[idx];
    }
    const Tensor3<T, dim2, dim3, dim4>& operator[](const int& idx) const {
        return data[idx];
    }
    std::array<T, dim1 * dim2 * dim3 * dim4>& Ravel(){
        union U{
            std::array<Tensor3<T, dim2, dim3, dim4>, dim1> data;
            std::array<T, dim1 * dim2 * dim3 * dim4> raveled;
        };
        return ((U*)&data)->raveled;
    }
    const std::array<T, dim1 * dim2 * dim3 * dim4>& Ravel() const {
        union U{
            std::array<Tensor3<T, dim2, dim3, dim4>, dim1> data;
            std::array<T, dim1 * dim2 * dim3 * dim4> raveled;
        };
        return ((U*)&data)->raveled;
    }
    void Fill(const T& fill_value) {
        fill(Ravel().begin(), Ravel().end(), fill_value);
    }
    template<int p0, int p1, int p2, int p3>
    auto Permute() const {
        // d0 が 1 の場合、もともと 1 次元目だったところが 0 次元目に来る
        constexpr static auto d = std::array<int, 4>{dim1, dim2, dim3, dim4};
        auto permuted = Tensor4<T, d[p0], d[p1], d[p2], d[p3]>();
        for(int i1=0; i1<dim1; i1++)
            for(int i2=0; i2<dim2; i2++)
                for(int i3=0; i3<dim3; i3++)
                    for(int i4=0; i4<dim4; i4++)
                        permuted
                            [std::array<int, 4>{i1,i2,i3,i4}[p0]]
                            [std::array<int, 4>{i1,i2,i3,i4}[p1]]
                            [std::array<int, 4>{i1,i2,i3,i4}[p2]]
                            [std::array<int, 4>{i1,i2,i3,i4}[p3]] = data[i1][i2][i3][i4];
        return permuted;
    }
    inline auto& operator+=(const Tensor4& rhs) {
        for(int i=0; i<dim1*dim2*dim3*dim4; i++) Ravel()[i] += rhs.Ravel()[i];
        return *this;
    }
    void Print() const {
        for(int i=0; i<dim1; i++){
            data[i].Print();
            if(i != dim1-1) std::cout << std::endl << std::endl;
        }
    }
};

namespace F{
    template<class Tensor>
    void Relu_(Tensor& input){
        for(auto&& value : input.Ravel()){
            value = std::max(value, (typename std::remove_reference<decltype(input.Ravel()[0])>::type)0);
        }
    }
    template<typename T, unsigned siz>
    void Relu_(std::array<T, siz>& input){
        for(auto&& value : input){
            value = max(value, (T)0);
        }
    }
    template<class T, size_t siz>
    void Softmax_(std::array<T, siz>& input){
        auto ma = std::numeric_limits<float>::min();
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
        std::array<float, n_features> weight;  // gamma
        std::array<float, n_features> bias;    // beta
        std::array<float, n_features> running_mean;
        std::array<float, n_features> running_var;
        Parameter() : weight(), bias(), running_mean(), running_var() {
            // weight と running_var は 1 で初期化
            std::fill(weight.begin(), weight.end(), 1.0f);
            std::fill(running_var.begin(), running_var.end(), 1.0f);
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
        std::array<float, output_dim> bias;
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

        #if false
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
                                //cout << input[in_channel][y_from][x_from] * parameters.weight[out_channel][in_channel][ky][kx] << " ";
                                output[out_channel][y][x] += input[in_channel][y_from][x_from] * parameters.weight[out_channel][in_channel][ky][kx];
                            }
                        }
                    }
                }
            }
        }
        #else
        // 効率化した版
        const auto permuted_input = input.template Permute<1, 2, 0>();
        //const auto permuted_weight = Tensor3<float, kernel_size, kernel_size, input_dim>();
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

        #endif
        batchnorm.Forward_(output);
    }
};

template<int in_features, int out_features>
struct Linear{
    struct Parameters{
        Matrix<float, out_features, in_features> weight;
        std::array<float, out_features> bias;
    } parameters;
    
    // コンストラクタ
    Linear() : parameters() {}

    void Forward(const std::array<float, in_features>& input, std::array<float, out_features>& output) const {
        output = parameters.bias;
        for(int out_channel=0; out_channel<out_features; out_channel++){
            for(int in_channel=0; in_channel<in_features; in_channel++){
                output[out_channel] += input[in_channel] * parameters.weight[out_channel][in_channel];
            }
        }
    }
};

template<int in_features=17, int n_blocks=12, int n_filters=32>
struct GeeseNet{
    TorusConv2d<in_features, n_filters> conv0;
    std::array<TorusConv2d<n_filters, n_filters>, n_blocks> blocks;
    Linear<n_filters, 4> head_policy;
    Linear<n_filters * 2, 1> head_value;
    
    // コンストラクタ
    GeeseNet() : conv0(), blocks(), head_policy(), head_value() {}

    template<int height, int width>
    void Forward(
        const Tensor3<float, in_features, height, width>& input,
        std::array<float, 4>& output_policy,
        float& output_value) const {
        
        static Tensor3<float, n_filters, height, width> h1, h2;
        conv0.Forward(input, h1);
        //for(int i=0; i<20; i++) cout << h1.Ravel()[i] << " \n"[i==19];
        F::Relu_(h1);
        //for(int i=0; i<20; i++) cout << h1.Ravel()[i] << " \n"[i==19];
        for(int idx_block=0; idx_block<n_blocks; idx_block++){
            if (idx_block%2 == 0){
                blocks[idx_block].Forward(h1, h2);
                h2 += h1;
                F::Relu_(h2);
            }else{
                blocks[idx_block].Forward(h2, h1);
                h1 += h2;
                F::Relu_(h1);
            }
        }
        if (n_blocks % 2 == 1) h1 = h2;
        static auto h_head = std::array<float, n_filters>();
        static auto h_head_avg = std::array<float, n_filters*2>();
        std::fill(h_head_avg.begin(), h_head_avg.end(), 0.0);
        for(int y=0; y<height; y++){
            for(int x=0; x<width; x++){
                // input[0] には、自エージェントの頭を表すフラグが入っている
                if(input[0][y][x] == 1.0f) {
                    for(int channel=0; channel<n_filters; channel++){
                        h_head[channel] = h1[channel][y][x];
                    }
                }
                for(int channel=0; channel<n_filters; channel++){
                    h_head_avg[n_filters + channel] += h1[channel][y][x];
                }
            }
        }
        for(int channel=0; channel<n_filters; channel++) {
            h_head_avg[channel] = h_head[channel];
            h_head_avg[n_filters + channel] /= (float)(height * width);
        }
        //for(auto&& hh : h_head) cout << hh << " ";
        //cout << endl;
        head_policy.Forward(h_head, output_policy);
        F::Softmax_(output_policy);

        head_value.Forward(h_head_avg, *(std::array<float, 1>*)&output_value);
        output_value = tanhf(output_value);
    }
};

namespace test{
    void CheckLinear(){
        auto linear = Linear<3, 4>();
        std::iota(linear.parameters.weight.Ravel().begin(), linear.parameters.bias.end(), 0.0f);
        auto input = std::array<float, 3>{3.0, -2.0, 1.0};
        auto output = std::array<float, 4>();
        linear.Forward(input, output);
        for(int i=0; i<4; i++) std::cout << output[i] << " \n"[i==3];  // => 12, 19, 26, 33
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
        std::cout << "Input:" << std::endl;
        input.Print();
        std::cout << "Output:" << std::endl;
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
    void CheckGeeseNet(){
        auto model = GeeseNet<>();
        auto input = Tensor3<float, 17, 7, 11>{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0077,0.0078,0.0079,0.007999999,0.0081,0.0082,0.0083,0.0084,0.0084999995,0.008599999,0.0087,0.0088,0.0089,0.009,0.009099999,0.0092,0.0093,0.0094,0.0095,0.0095999995,0.009699999,0.0098,0.0099,0.01,0.0101,0.010199999,0.0103,0.0104,0.0105,0.0106,0.0106999995,0.010799999,0.0109,0.011,0.0111,0.0112,0.011299999,0.011399999,0.0115,0.0116,0.0117,0.0117999995,0.011899999,0.012,0.0121,0.0122,0.0123,0.012399999,0.012499999,0.0126,0.0127,0.0128,0.0128999995,0.012999999,0.0131,0.0132,0.0133,0.0134,0.013499999,0.013599999,0.0137,0.0138,0.0139,0.0139999995,0.014099999,0.0142,0.0143,0.0144,0.0145,0.014599999,0.014699999,0.0148,0.0149,0.015,0.0150999995,0.015199999,0.015299999,0.0154,0.0155,0.0156,0.0157,0.0158,0.015899999,0.015999999,0.016099999,0.0162,0.0163,0.0164,0.0165,0.0166,0.0167,0.0168,0.0169,0.016999999,0.017099999,0.017199999,0.0173,0.0174,0.0175,0.0176,0.0177,0.0178,0.0179,0.018,0.018099999,0.018199999,0.018299999,0.0184,0.0185,0.0186,0.0187,0.0188,0.0189,0.019,0.0191,0.019199999,0.019299999,0.019399999,0.0195,0.0196,0.0197,0.0198,0.0199,0.02,0.0201,0.0202,0.020299999,0.020399999,0.020499999,0.0206,0.0207,0.0208,0.0209,0.021,0.0211,0.0212,0.0213,0.021399999,0.021499999,0.021599999,0.021699999,0.0218,0.0219,0.022,0.0221,0.0222,0.0223,0.0224,0.022499999,0.022599999,0.022699999,0.022799999,0.0229,0.023,0.0231,0.0232,0.0233,0.0234,0.0235,0.023599999,0.023699999,0.023799999,0.023899999,0.024,0.0241,0.0242,0.0243,0.0244,0.0245,0.0246,0.024699999,0.024799999,0.024899999,0.024999999,0.0251,0.0252,0.0253,0.0254,0.0255,0.0256,0.0257,0.025799999,0.025899999,0.025999999,0.026099999,0.0262,0.0263,0.0264,0.0265,0.0266,0.0267,0.0268,0.026899999,0.026999999,0.027099999,0.027199998,0.0273,0.0274,0.0275,0.0276,0.0277,0.0278,0.0279,0.027999999,0.028099999,0.028199999,0.028299998,0.0284,0.0285,0.0286,0.0287,0.0288,0.0289,0.029,0.029099999,0.029199999,0.029299999,0.029399998,0.0295,0.0296,0.0297,0.0298,0.0299,0.03,0.0301,0.030199999,0.030299999,0.030399999,0.030499998,0.030599998,0.0307,0.0308,0.0309,0.031,0.0311,0.0312,0.0313,0.0314,0.0315,0.0316,0.0317,0.031799998,0.0319,0.031999998,0.0321,0.032199997,0.0323,0.0324,0.0325,0.0326,0.0327,0.0328,0.032899998,0.033,0.033099998,0.0332,0.033299997,0.0334,0.0335,0.0336,0.0337,0.0338,0.0339,0.033999998,0.0341,0.034199998,0.0343,0.034399997,0.0345,0.0346,0.0347,0.0348,0.0349,0.035,0.035099998,0.0352,0.035299998,0.0354,0.035499997,0.0356,0.0357,0.0358,0.0359,0.036,0.0361,0.036199998,0.0363,0.036399998,0.0365,0.036599997,0.0367,0.0368,0.0369,0.037,0.0371,0.0372,0.037299998,0.0374,0.037499998,0.0376,0.037699997,0.0378,0.0379,0.038,0.0381,0.0382,0.0383,0.038399998,0.0385,0.038599998,0.0387,0.038799997,0.0389,0.039,0.0391,0.0392,0.0393,0.0394,0.039499998,0.0396,0.039699998,0.0398,0.039899997,0.04,0.0401,0.0402,0.0403,0.0404,0.0405,0.040599998,0.0407,0.040799998,0.0409,0.040999997,0.0411,0.0412,0.0413,0.0414,0.0415,0.0416,0.041699998,0.0418,0.041899998,0.042,0.042099997,0.0422,0.0423,0.0424,0.0425,0.0426,0.0427,0.042799998,0.0429,0.042999998,0.0431,0.043199997,0.0433,0.043399997,0.0435,0.0436,0.0437,0.0438,0.043899998,0.044,0.044099998,0.0442,0.044299997,0.0444,0.044499997,0.0446,0.0447,0.0448,0.0449,0.044999998,0.0451,0.045199998,0.0453,0.045399997,0.0455,0.045599997,0.0457,0.0458,0.0459,0.046,0.046099998,0.0462,0.046299998,0.0464,0.046499997,0.0466,0.046699997,0.0468,0.0469,0.047,0.0471,0.047199998,0.0473,0.047399998,0.0475,0.047599997,0.0477,0.047799997,0.0479,0.048,0.0481,0.0482,0.048299998,0.0484,0.048499998,0.0486,0.048699997,0.0488,0.048899997,0.049,0.0491,0.0492,0.0493,0.049399998,0.0495,0.049599998,0.0497,0.049799997,0.0499,0.049999997,0.0501,0.0502,0.0503,0.0504,0.050499998,0.0506,0.050699998,0.0508,0.050899997,0.051,0.051099997,0.0512,0.0513,0.0514,0.0515,0.051599998,0.0517,0.051799998,0.0519,0.051999997,0.0521,0.052199997,0.0523,0.0524,0.0525,0.0526,0.052699998,0.0528,0.052899998,0.053,0.053099997,0.0532,0.053299997,0.0534,0.0535,0.0536,0.0537,0.053799998,0.0539,0.053999998,0.0541,0.054199997,0.0543,0.054399997,0.0545,0.0546,0.0547,0.0548,0.054899998,0.055,0.055099998,0.0552,0.055299997,0.0554,0.055499997,0.0556,0.0557,0.0558,0.0559,0.055999998,0.0561,0.056199998,0.0563,0.056399997,0.0565,0.056599997,0.0567,0.0568,0.0569,0.057,0.057099998,0.0572,0.057299998,0.0574,0.057499997,0.0576,0.057699997,0.0578,0.0579,0.058,0.0581,0.058199998,0.0583,0.058399998,0.0585,0.058599997,0.0587,0.058799997,0.0589,0.059,0.0591,0.0592,0.059299998,0.0594,0.059499998,0.0596,0.059699997,0.0598,0.059899997,0.06,0.060099997,0.0602,0.0603,0.060399998,0.0605,0.060599998,0.0607,0.060799997,0.0609,0.060999997,0.0611,0.061199997,0.0613,0.0614,0.061499998,0.0616,0.061699998,0.0618,0.061899997,0.062,0.062099997,0.0622,0.062299997,0.0624,0.0625,0.0626,0.062699996,0.0628,0.0629,0.063,0.063099995,0.0632,0.0633,0.0634,0.0635,0.063599996,0.0637,0.0638,0.0639,0.063999996,0.0641,0.0642,0.0643,0.064399995,0.0645,0.0646,0.0647,0.0648,0.064899996,0.065,0.0651,0.0652,0.065299995,0.0654,0.0655,0.0656,0.0657,0.065799996,0.0659,0.066,0.0661,0.066199996,0.0663,0.0664,0.0665,0.066599995,0.0667,0.0668,0.0669,0.067,0.067099996,0.0672,0.0673,0.0674,0.067499995,0.0676,0.0677,0.0678,0.0679,0.067999996,0.0681,0.0682,0.0683,0.068399996,0.0685,0.0686,0.0687,0.068799995,0.0689,0.069,0.0691,0.0692,0.069299996,0.0694,0.0695,0.0696,0.069699995,0.0698,0.0699,0.07,0.070099995,0.070199996,0.0703,0.0704,0.0705,0.070599996,0.0707,0.0708,0.0709,0.070999995,0.0711,0.0712,0.0713,0.0714,0.071499996,0.0716,0.0717,0.0718,0.071899995,0.072,0.0721,0.0722,0.072299995,0.072399996,0.0725,0.0726,0.0727,0.072799996,0.0729,0.073,0.0731,0.073199995,0.0733,0.0734,0.0735,0.0736,0.073699996,0.0738,0.0739,0.074,0.074099995,0.0742,0.0743,0.0744,0.074499995,0.074599996,0.0747,0.0748,0.0749,0.074999996,0.0751,0.0752,0.0753,0.075399995,0.0755,0.0756,0.0757,0.0758,0.075899996,0.076,0.0761,0.0762,0.076299995,0.0764,0.0765,0.0766,0.076699995,0.076799996,0.0769,0.077,0.0771,0.077199996,0.0773,0.0774,0.0775,0.077599995,0.0777,0.0778,0.0779,0.078,0.078099996,0.0782,0.0783,0.0784,0.078499995,0.0786,0.0787,0.0788,0.078899994,0.078999996,0.0791,0.0792,0.0793,0.079399996,0.0795,0.0796,0.0797,0.079799995,0.0799,0.08,0.0801,0.0802,0.080299996,0.0804,0.0805,0.0806,0.080699995,0.0808,0.0809,0.081,0.081099994,0.081199996,0.0813,0.0814,0.0815,0.081599995,0.0817,0.0818,0.0819,0.081999995,0.0821,0.0822,0.0823,0.0824,0.082499996,0.0826,0.0827,0.0828,0.082899995,0.083,0.0831,0.0832,0.083299994,0.083399996,0.0835,0.0836,0.0837,0.083799995,0.0839,0.084,0.0841,0.084199995,0.0843,0.0844,0.0845,0.0846,0.084699996,0.0848,0.0849,0.085,0.085099995,0.0852,0.0853,0.0854,0.085499994,0.085599996,0.0857,0.0858,0.0859,0.085999995,0.0861,0.0862,0.0863,0.086399995,0.0865,0.0866,0.0867,0.086799994,0.086899996,0.087,0.0871,0.0872,0.087299995,0.0874,0.0875,0.0876,0.087699994,0.087799996,0.0879,0.088,0.0881,0.088199995,0.0883,0.0884,0.0885,0.088599995,0.0887,0.0888,0.0889,0.088999994,0.089099996,0.0892,0.0893,0.0894,0.089499995,0.0896,0.0897,0.0898,0.089899994,0.089999996,0.0901,0.0902,0.0903,0.090399995,0.0905,0.0906,0.0907,0.090799995,0.0909,0.091,0.0911,0.091199994,0.091299996,0.0914,0.0915,0.0916,0.091699995,0.0918,0.0919,0.092,0.092099994,0.092199996,0.0923,0.0924,0.0925,0.092599995,0.0927,0.0928,0.0929,0.092999995,0.0931,0.0932,0.0933,0.093399994,0.093499996,0.0936,0.0937,0.0938,0.093899995,0.094,0.0941,0.0942,0.094299994,0.094399996,0.0945,0.0946,0.0947,0.094799995,0.0949,0.095,0.0951,0.095199995,0.0953,0.0954,0.0955,0.095599994,0.095699996,0.0958,0.0959,0.096,0.096099995,0.0962,0.0963,0.0964,0.096499994,0.096599996,0.0967,0.0968,0.0969,0.096999995,0.0971,0.0972,0.0973,0.097399995,0.0975,0.0976,0.0977,0.097799994,0.097899996,0.098,0.0981,0.0982,0.098299995,0.0984,0.0985,0.0986,0.098699994,0.098799996,0.0989,0.099,0.0991,0.099199995,0.0993,0.0994,0.0995,0.099599995,0.0997,0.0998,0.0999,0.099999994,0.100099996,0.1002,0.1003,0.1004,0.100499995,0.1006,0.1007,0.1008,0.100899994,0.100999996,0.1011,0.1012,0.1013,0.101399995,0.1015,0.1016,0.1017,0.101799995,0.1019,0.102,0.1021,0.102199994,0.102299996,0.1024,0.1025,0.1026,0.102699995,0.1028,0.1029,0.103,0.103099994,0.103199996,0.1033,0.1034,0.10349999,0.103599995,0.1037,0.1038,0.1039,0.103999995,0.1041,0.1042,0.1043,0.104399994,0.104499996,0.1046,0.1047,0.1048,0.104899995,0.105,0.1051,0.1052,0.105299994,0.105399996,0.1055,0.1056,0.10569999,0.105799995,0.1059,0.106,0.1061,0.106199995,0.1063,0.1064,0.1065,0.106599994,0.106699996,0.1068,0.1069,0.107,0.107099995,0.1072,0.1073,0.1074,0.107499994,0.107599996,0.1077,0.1078,0.10789999,0.107999995,0.1081,0.1082,0.1083,0.108399995,0.1085,0.1086,0.1087,0.108799994,0.108899996,0.109,0.1091,0.1092,0.109299995,0.1094,0.1095,0.1096,0.109699994,0.109799996,0.1099,0.11,0.11009999,0.110199995,0.1103,0.1104,0.1105,0.110599995,0.1107,0.1108,0.1109,0.110999994,0.111099996,0.1112,0.1113,0.1114,0.111499995,0.1116,0.1117,0.1118,0.111899994,0.111999996,0.1121,0.1122,0.11229999,0.112399995,0.1125,0.1126,0.1127,0.112799995,0.1129,0.113,0.1131,0.113199994,0.113299996,0.1134,0.1135,0.1136,0.113699995,0.1138,0.1139,0.114,0.114099994,0.114199996,0.1143,0.1144,0.11449999,0.114599995,0.1147,0.1148,0.1149,0.114999995,0.1151,0.1152,0.1153,0.115399994,0.115499996,0.1156,0.1157,0.1158,0.115899995,0.116,0.1161,0.1162,0.116299994,0.116399996,0.1165,0.1166,0.11669999,0.116799995,0.1169,0.117,0.1171,0.117199995,0.1173,0.1174,0.1175,0.117599994,0.117699996,0.1178,0.1179,0.118,0.118099995,0.1182,0.1183,0.1184,0.118499994,0.118599996,0.1187,0.1188,0.11889999,0.118999995,0.1191,0.1192,0.1193,0.119399995,0.1195,0.1196,0.1197,0.119799994,0.119899996,0.12,0.1201,0.12019999,0.120299995,0.1204,0.1205,0.1206,0.120699994,0.120799996,0.1209,0.121,0.12109999,0.121199995,0.1213,0.1214,0.1215,0.121599995,0.1217,0.1218,0.1219,0.121999994,0.122099996,0.1222,0.1223,0.12239999,0.122499995,0.1226,0.1227,0.1228,0.122899994,0.122999996,0.1231,0.1232,0.12329999,0.123399995,0.1235,0.1236,0.1237,0.123799995,0.1239,0.124,0.1241,0.124199994,0.124299996,0.1244,0.1245,0.12459999,0.124699995,0.1248,0.1249,0.125,0.1251,0.1252,0.12529999,0.12539999,0.1255,0.1256,0.1257,0.1258,0.1259,0.126,0.1261,0.12619999,0.12629999,0.1264,0.1265,0.1266,0.1267,0.1268,0.1269,0.127,0.12709999,0.12719999,0.1273,0.1274,0.1275,0.1276,0.1277,0.1278,0.12789999,0.12799999,0.1281,0.1282,0.1283,0.1284,0.1285,0.1286,0.1287,0.12879999,0.12889999,0.129,0.1291,0.1292,0.1293,0.1294,0.1295,0.1296,0.12969999,0.12979999,0.1299,0.13,0.1301,0.1302,0.1303,0.1304,0.1305,0.13059999,0.13069999,0.1308};
        auto output_policy = std::array<float, 4>();
        auto output_value = 0.0f;
        // この include がコンパイル時に激重なので必要なければコメントアウトする
        //#include "parameters.hpp"
        
        model.Forward(input, output_policy, output_value);
        for(int i=0; i<4; i++) std::cout << output_policy[i] << " \n"[i==3];
        std::cout << output_value << std::endl;
    }
    // 100 回予測する時間の計測
    void CheckGeeseNetTime(){
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
            for(int j=0; j<4; j++) std::cout << output_policy[j] << " \n"[j==3];
            std::cout << output_value << "\n";
        }
        std::cout << "time=" << time() - t0 << std::endl;
    }
};
    
struct Evaluator{
    GeeseNet<> model;
    Evaluator(){
        // パラメータ設定
        #include "EvalParameters.hpp"
    }
    template<class T, int max_size>
    using Stack = Stack<T, max_size>;
    auto evaluate(const std::array<Stack<hungry_geese::Cpoint, 77>, 4>& geese, const std::array<hungry_geese::Cpoint, 2>& foods) const {
        static auto input = Tensor3<float, 17, 7, 11>();
        input.Fill(0.0);
        for(int agent=0; agent<4; agent++){
            const auto& goose = geese[agent];
            if(goose.size() > 0){
                // 頭
                input[0+agent][goose.front().X()][goose.front().Y()] = 1.0f;
                // しっぽ
                input[4+agent][goose.back().X()][goose.back().Y()] = 1.0f;
                // 全部
                for(const auto& pos : goose)
                    input[8+agent][pos.X()][pos.Y()] = 1.0f;
                // 1 つ前のターンの頭 (元の実装とちょっと違う)
                if(goose.size() >= 2)
                    input[12+agent][goose[1].X()][goose[1].Y()] = 1.0f;
            }
        }
        // 食べ物
        for(const auto& pos : foods) input[16][pos.X()][pos.Y()] = 1.0f;
        struct{
            std::array<float, 4> policy;
            float value;
        } result;
        model.Forward(input, result.policy, result.value);
        std::swap(result.policy[1],result.policy[2]);
        std::swap(result.policy[1],result.policy[3]);
        return result;
    }
};

}

namespace hungry_geese {

// Stage
struct Stage {
    float remaining_overage_time;
    int current_step;
    std::array<nagiss_library::Stack<Cpoint, 77>, 4> geese;
    std::array<Cpoint, 2> foods;
    std::array<int, 4> last_actions;
};

// DUCT
struct Duct {
    Duct() : node_buffer(), children_buffer(), model(), t_sum() {}
    constexpr static int node_buffer_size = 10000;
    constexpr static int children_buffer_size = 100000;
    // NodeType
    enum struct NodeType : bool {
        AGENT_NODE,
        FOOD_NODE,
    };
    // State
    struct State {
        State() : geese(), boundary(), foods(), current_step(), last_actions(), ranking() {}
        State(hungry_geese::Stage aStage, int aIndex) : geese(), boundary(), foods(), current_step(), last_actions(), ranking() {
            int index = 0;
            for (int i = 0; i < 4; ++i) {
                boundary[i] = index;
                for (int j = 0; j < aStage.geese[i].size(); ++j) {
                    geese[index] = Cpoint(aStage.geese[i][j]);
                    index++;
                }
            }
            boundary[4] = index;
            for (int i = 0; i < 2; ++i) {
                foods[i] = Cpoint(aStage.foods[i]);
            }
            current_step = aStage.current_step;
            for (int i = 0; i < 4; ++i) {
                last_actions += (1 << (i + i)) * aStage.last_actions[i];
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
        std::array<Cpoint,77> geese;
        std::array<signed char, 5> boundary;
        std::array<Cpoint, 2> foods;
        unsigned char current_step; // ターン数
        unsigned char last_actions; // [0,256)の整数で手を表現する
        std::array<signed char, 4> ranking;

        // idx_agent 番目のgooseのサイズを返す
        unsigned char goose_size(unsigned char idx_agent) {
            return boundary[idx_agent + 1] - boundary[idx_agent];
        }
        // 手を引数で渡して次のノードを得る
        // food_sub : 食べ物が2個同時に消える場合256だと情報足りない
        State NextState(NodeType node_type, const unsigned char agent_action, const unsigned char food_sub) const {
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
        // シミュレート
        static void Simulate(State &state, unsigned char agent_action) {
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
                auto head = Translate(state.geese[state.boundary[i]], agent_action%4);
                agent_action /= 4;
                bool eatFood = false;
                for (int j = 0; j < 2; ++j) {
                    if (head == state.foods[j]) {
                        eatFood = true;
                        state.foods[j] = Cpoint(-1);
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
                if ((state.current_step + 1) % 40 == 0) {
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
        // 終局状態か(プレイヤー0が生存しているか)
        bool Finished() const {
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
    };

    // Node
    struct Node {
        State state; // 状態
        std::array<std::array<float, 4>, 4> policy; // 各エージェントの方策
        std::array<float, 4> value; // 状態評価値4人分
        std::array<std::array<float, 4>, 4> worth; // 各エージェント視点の各手の累積価値
        std::array<std::array<int, 4>, 4> n; // 各手の選ばれた回数
        int n_children; // 子ノードの数
        int children_offset; // 子ノード
        NodeType node_type; // エージェントのノード (0) か、食べ物のノード (1) か

        // 問い合わせ
        const std::array<std::array<float, 4>, 4>& GetPolicy() const {
            return policy;
        }
        const std::array<std::array<float, 4>, 4>& GetWorth() const {
            return worth;
        }

        Node() : state(), policy(), value(), worth(), n(), n_children(), children_offset(), node_type() {}

        Node(const State& aState, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer) : state(aState), policy(), value(), worth(), n(), children_offset(), node_type() {
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

        bool Expanded() const {// 既にモデルによって評価されているかを返す
            return (policy[0][0] != -100.0);
        }
        // アーク評価値
        float Argvalue(const int& idx_agent, const int& idx_move, const int& t_sum) {
            constexpr float c_puct = 1.0;
            return GetWorth()[idx_agent][idx_move] / (float)(1e-1 + n[idx_agent][idx_move]) + c_puct * GetPolicy()[idx_agent][idx_move] * std::sqrt(t_sum) / (float)(1 + n[idx_agent][idx_move]);
        }
        // 例の式を使って (食べ物のノードの場合はランダムに) 手を選ぶ  // 手の全パターンをひとつの値で表す。全員が 3 方向に移動できるなら 0 から 80 までの値をとる。Move メソッドで具体的な手に変換できる
        int ChooseMove(const int& t_sum) {
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
        // k 番目の行動によって遷移する子ノードを返す 
        // その子ノードが初めて遷移するものだった場合、新たに領域を確保してノードを作る
        Node& KthChildren(nagiss_library::Stack<Node, node_buffer_size>& node_buffer, nagiss_library::Stack<Node*, children_buffer_size>& children_buffer, const int& k) {
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
    };

    nagiss_library::Stack<Node, 10000> node_buffer;
    nagiss_library::Stack<Node*, 100000> children_buffer;
    evaluation_function::Evaluator model;
    int t_sum; // 累計試行回数

    // 初期化
    void InitDuct(const Node& arg_state) {
        node_buffer.clear();
        node_buffer.push(arg_state);
        children_buffer.clear();
        t_sum = 0;
    }
    void InitDuct(hungry_geese::Stage aStage, int aIndex) {
        children_buffer.clear();
        t_sum = 0;
        auto state = Duct::State(aStage, aIndex);
        auto node = Duct::Node(state, children_buffer);
        node_buffer.clear();
        node_buffer.push(node);
    }

    // 探索
    void Search(const float timelimit) {
        double timebegin = nagiss_library::time();
        while (nagiss_library::time() - timebegin < timelimit) {
            Iterate();
            t_sum++;
            if (t_sum >=3)break;
        }
    }
    Node& RootNode() {
        return node_buffer[0];
    }
    void Iterate() {
        // 根から葉に移動
        Node* v = &RootNode();
        nagiss_library::Stack<int, 100> path;
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
            std::array<nagiss_library::Stack<Cpoint, 77>, 4> geese;
            std::array<Cpoint, 2> foods;
            for (int i = 0; i < 4; ++i) {
                for (int j = v->state.boundary[i]; j < v->state.boundary[i + 1]; ++j) {
                    geese[i].push(v->state.geese[j]);
                }
            }
            for (int i = 0; i < 2; ++i) {
                foods[i] = v->state.foods[i];
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

    // いつもの
    static constexpr std::array<int, 4> dx = {-1, 0, 1, 0};
    static constexpr std::array<int, 4> dy = {0, 1, 0, -1};

    static Cpoint Translate(Cpoint aPos, unsigned char Direction) {
        int nx = aPos.X();
        int ny = aPos.Y();
        // nx += dx[Direction];
        if (nx < 0) {
            nx += rows; 
        }
        if (nx == rows) {
            nx = 0;
        }
        // ny += dy[Direction];
        if (ny < 0) {
            ny += columns; 
        }
        if (ny == columns) {
            ny = 0;
        }
        return Cpoint(nx,ny);
    }
};

// Agent
struct MctsAgent {
    std::array<Stage, 200> stages;
    Duct duct;
    evaluation_function::Evaluator model;
    int GetDirection(Cpoint pre, Cpoint cur) {
        int pre_x = pre.X(), pre_y = pre.Y();
        int cur_x = cur.X(), cur_y = cur.Y();
        if ((pre_x - 1 + rows) % rows == cur_x) return 0;
        else if ((pre_y + 1) % columns == cur_y) return 1;
        else if ((pre_x + 1) % rows == cur_x) return 2;
        else return 3;
    }
    void input(int idx) {
        auto& stage = stages[idx];
        std::cin >> stage.remaining_overage_time;
        std::cin >> stage.current_step;
        for (int i = 0; i < 4; ++i) {
            int n; std::cin >> n;
            for (int j = 0; j < n; ++j) {
                int g; std::cin >> g;
                stage.geese[i].push(Cpoint(g));
            }
        }
        for (int i = 0; i < 2; ++i) {
            int food; std::cin >> food;
            stage.foods[i] = Cpoint(food);
        }
        if (idx == 0) {
            for (int i = 0; i < 4; ++i) {
                stage.last_actions[i] = -1;
            }
        }
        else {
            for (int i = 0; i < 4; ++i) {
                if (stage.geese[i].size() == 0) {
                    stage.last_actions[i] = 0;
                }
                else {
                    stage.last_actions[i] = GetDirection(stages[idx - 1].geese[i][0], stage.geese[i][0]);
                }
            }
        }
    }
    void WriteAns(int direction) {
        if (direction == 0) {
            std::cout << "move l" << std::endl;
        }
        else if (direction == 1) {
            std::cout << "move d" << std::endl;
        }
        else if (direction == 2) {
            std::cout << "move r" << std::endl;
        }
        else {
            std::cout << "move u" << std::endl;
        }
    }
    void solve() {
        for (int turn = 0; turn < 200; ++turn) {
            input(turn);
            auto stage = stages[turn];
            if (turn == 0) {
                auto res = model.evaluate(stage.geese, stage.foods);
                int opt_action = 0;
                for (int i = 0; i < 4; ++i) {
                    if (res.policy[opt_action] < res.policy[i]) {
                        opt_action = i;
                    }
                }
                WriteAns(opt_action);
            }
            else {
                duct.InitDuct(stage, 0);
                duct.Search(0.6);
                std::array<float, 4> policy;
                auto rootnode = duct.RootNode();
                for (int i = 0; i < 4; ++i) {
                    policy[i] = (float)rootnode.n[0][i] / (float)(rootnode.n[0][0] + rootnode.n[0][1] + rootnode.n[0][2] + rootnode.n[0][3]);
                }
                int opt_action = 0;
                for (int i = 0; i < 4; ++i) {
                    if (policy[opt_action] < policy[i]) {
                        opt_action = i;
                    }
                }
                WriteAns(opt_action);
            }
        }
    }
};

}

int main() {
    hungry_geese::MctsAgent Agent;
    Agent.solve();
}
