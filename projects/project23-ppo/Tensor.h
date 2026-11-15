#pragma once;

struct Tensor {
    float * data;
    int     count;
    int     dim;
};

void tn_mul( float scalar, Tensor & tn );
void tn_add( Tensor & a, Tensor & b );