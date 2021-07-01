#pragma once

#include <cassert>
#include <cstdio>

// メッセージ付きアサート
// aExp 条件式
// ... 書式。引数の設定方法は std::printf に準ずる
#define ASSERT_MSG(aExp, ...) \
    do { \
        if (!(aExp)) { \
            std::printf(__VA_ARGS__); \
            std::printf("\n"); \
            assert(aExp); \
        } \
    } while (false)

// 標準のアサート
// 条件式が true になることを表明する
// 条件式が true にならなかった場合、既定のメッセージを表示して停止させる
// aExp 条件式
#define ASSERT(aExp) ASSERT_MSG(aExp, "Assertion Failed")

// 整数限定
#define ASSERT_RANGE(value, left, right) \
    ASSERT_MSG((left <= value) && (value < right), \
        "`%s` (%d) is out of range [%d, %d)", #value, value, left, right)
