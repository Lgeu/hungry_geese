#pragma once
#include "Point.hpp"
#include "Parameter.hpp"

namespace hungry_geese {

Point::Point() : x(), y(), id() {}
    
Point::Point(int aX, int aY) : x(aX), y(aY) {
    id = Parameter::columns * aX + aY;
}

Point::Point(int aId): x(), y(), id(aId) {
    x = aId / Parameter::columns;
    y = aId % Parameter::columns;
}

bool Point::operator== (const Point &aPos) const {
    return (id == aPos.id);
}

}