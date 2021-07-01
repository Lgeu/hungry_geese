#pragma once

namespace hungry_geese {
    
// Agentの行動
enum Action {
    NORTH,
    EAST,
    SOUTH,
    WEST,
    WAIT, // 何もしない行動はないが一応加えておく
};

}