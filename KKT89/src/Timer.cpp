#pragma once
#include "Timer.hpp"

namespace hungry_geese {

Timer::Timer(): mTimeBegin(std::clock_t()), mTimeEnd(std::clock_t()) {
}

void Timer::start() {
    mTimeBegin = ::std::clock();
}

void Timer::stop() {
    mTimeEnd = ::std::clock();
}

float Timer::elapsedSec() const {
    return static_cast<float>(mTimeEnd - mTimeBegin) / CLOCKS_PER_SEC;
}

}