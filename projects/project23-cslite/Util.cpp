#include "Util.h"

#include <random>

static std::mt19937 & util_rng() {
    static std::mt19937 rng( 1234u );
    return rng;
}

float random_uniform() {
    std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
    return dist( util_rng() );
}

float random_uniform( float min, float max ) {
    std::uniform_real_distribution<float> dist( min, max );
    return dist( util_rng() );
}

float random_normal() {
    std::normal_distribution<float> dist( 0.0f, 1.0f );
    return dist( util_rng() );
}

// Returns an int in [0, max).
int random_int( int max ) {
    std::uniform_int_distribution<int> dist( 0, max - 1 );
    return dist( util_rng() );
}
