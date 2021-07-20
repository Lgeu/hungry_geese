#pragma once
#include "Point.hpp"
#include "Parameter.hpp"

namespace hungry_geese {

Cpoint::Cpoint() : mC() {}

Cpoint::Cpoint(int aX, int aY) {
    mC = aX * hungry_geese::Parameter::columns + aY;
}

Cpoint::Cpoint(int aId) {
    mC = aId;
}

int Cpoint::X() const {
    return (int)mC / Parameter::columns;
}

int Cpoint::Y() const {
    return (int)mC % Parameter::columns;
}

int Cpoint::Id() const {
    return (int)mC;
}

Cpoint& Cpoint::operator= (const Cpoint &aPos) {
    mC = aPos.Id();
    return *this;
}

bool Cpoint::operator== (const Cpoint &aPos) const {
    return (mC == aPos.Id());
}

}