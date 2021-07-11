#pragma once

#include <cassert>
#include <cstdio>

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
    