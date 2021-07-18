#include "library.hpp"

namespace feature {

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
                std::cout << ((lo >> y * 11 + x) & 1);
            }
            std::cout << std::endl;
        }
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 11; x++) {
                std::cout << ((hi >> y * 11 + x) & 1);
            }
            std::cout << std::endl;
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
    inline int Neighbor(const int& idx, const std::array<nagiss_library::Vec2<int>, 12>& dyxs) const {
        int res = 0;
        auto yx0 = nagiss_library::Vec2<int>{ idx / 11, idx % 11 };
        for (int i = 0; i < (int)12; i++) {
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
        std::array<nagiss_library::Vec2<int>, 12> dyxs ={nagiss_library::Vec2<int>{-2, 0}, { -1,-1 }, { -1,0 }, { -1,1 }, { 0,-2 }, { 0,-1 }, { 0,1 }, { 0,2 }, { 1,-1 }, { 1,0 }, { 1,1 }, { 2,0 }};
        return Neighbor(idx, dyxs);
    }
    inline int Neighbor(const int& idx, const std::array<nagiss_library::Vec2<int>, 7>& dyxs) const {
        int res = 0;
        auto yx0 = nagiss_library::Vec2<int>{ idx / 11, idx % 11 };
        for (int i = 0; i < (int)7; i++) {
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
    inline int NeighborUp7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = std::array<Vec2<int>, 7>{Vec2<int>{-3, 0}, { -2,-1 }, { -2,0 }, { -2,1 }, { -1,-1 }, { -1,0 }, { -1,1 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborDown7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = std::array<Vec2<int>, 7>{Vec2<int>{3, 0}, { 2,-1 }, { 2,0 }, { 2,1 }, { 1,-1 }, { 1,0 }, { 1,1 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborLeft7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = std::array<Vec2<int>, 7>{Vec2<int>{0, -3}, { -1,-2 }, { 0,-2 }, { 1,-2 }, { -1,-1 }, { 0,-1 }, { 1,-1 }};
        return Neighbor(idx, dyxs);
    }
    inline int NeighborRight7(const int& idx) const {  // 遅かったら表引き->pext->表引きで高速化する
        using nagiss_library::Vec2;
        constexpr auto dyxs = std::array<Vec2<int>, 7>{Vec2<int>{0, 3}, { -1,2 }, { 0,2 }, { 1,2 }, { -1,1 }, { 0,1 }, { 1,1 }};
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

}
