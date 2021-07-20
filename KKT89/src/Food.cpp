#pragma once
#include "Food.hpp"

namespace hungry_geese {

Food::Food() : mPosition() {
    mIsEaten = false;
}

Food::Food(Cpoint aPos) : mPosition(aPos) {
    mIsEaten = false;
}

const Cpoint& Food::pos() const {
    return mPosition;
}

bool Food::isEaten() const {
    return mIsEaten;
}

void Food::setIsEaten(bool isEaten) {
    mIsEaten = isEaten;
}

}