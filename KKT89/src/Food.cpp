#pragma once
#include "Food.hpp"

namespace hungry_geese {

Food::Food() : mPosition() {
    mIsEaten = false;
}

Food::Food(Point aPos) : mPosition(aPos) {
    mIsEaten = false;
}

const Point& Food::pos() const {
    return mPosition;
}

bool Food::isEaten() const {
    return mIsEaten;
}

void Food::setIsEaten(bool isEaten) {
    mIsEaten = isEaten;
}

}