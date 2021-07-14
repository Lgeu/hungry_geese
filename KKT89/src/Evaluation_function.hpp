#pragma once
#include "library.hpp"
#include "Point.hpp"
#include <iostream>
#include <numeric>
#include <array>
#include <math.h>
#include <limits>
#include <time.h>

namespace hungry_geese {

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
        // #include "EvalParameters.hpp"
    }
    template<class T, int max_size>
    using Stack = Stack<T, max_size>;
    auto evaluate(const std::array<Stack<Point, 77>, 4>& geese, const std::array<Point, 2>& foods) const {
        static auto input = Tensor3<float, 17, 7, 11>();
        input.Fill(0.0);
        for(int agent=0; agent<4; agent++){
            const auto& goose = geese[agent];
            if(goose.size() > 0){
                // 頭
                input[0+agent][goose.front().x][goose.front().y] = 1.0f;
                // しっぽ
                input[4+agent][goose.back().x][goose.back().y] = 1.0f;
                // 全部
                for(const auto& pos : goose)
                    input[8+agent][pos.x][pos.y] = 1.0f;
                // 1 つ前のターンの頭 (元の実装とちょっと違う)
                if(goose.size() >= 2)
                    input[12+agent][goose[1].x][goose[1].y] = 1.0f;
            }
        }
        // 食べ物
        for(const auto& pos : foods) input[16][pos.x][pos.y] = 1.0f;
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
