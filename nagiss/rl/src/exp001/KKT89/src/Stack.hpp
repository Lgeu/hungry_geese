#pragma once
#include <array>
#include "Assert.hpp"

namespace hungry_geese {

// nagiss/library.hpp
template<class T,int max_size>
struct Stack {
    std::array<T, max_size> data;
    int right;
    inline Stack() : data(), right(0) {}
    inline Stack(const int n) : data(), right(0) { resize(n); }
    inline Stack(const int n, const T& val) : data(), right(0) { resize(n, val); }
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
};

}