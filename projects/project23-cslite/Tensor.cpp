#include "Tensor.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

static std::mt19937 & tn_rng() {
    static std::mt19937 rng( 1234u );
    return rng;
}

// ---- Lifetime --------------------------------------------------------------

Tensor tn_create_shape( int dim, const int * shape ) {
    assert( dim > 0 && dim <= TN_MAX_DIMS );
    Tensor tn;
    tn.dim = dim;
    tn.count = 1;
    for ( int i = 0; i < dim; i++ ) {
        assert( shape[i] > 0 );
        tn.shape[i] = shape[i];
        tn.count *= shape[i];
    }
    tn.data = (float *)calloc( tn.count, sizeof( float ) );
    assert( tn.data != nullptr );
    return tn;
}

Tensor tn_create( int n ) {
    int shape[1] = { n };
    return tn_create_shape( 1, shape );
}

Tensor tn_create( int rows, int cols ) {
    int shape[2] = { rows, cols };
    return tn_create_shape( 2, shape );
}

Tensor tn_clone( const Tensor & tn ) {
    Tensor out = tn_create_shape( tn.dim, tn.shape );
    memcpy( out.data, tn.data, sizeof( float ) * tn.count );
    return out;
}

void tn_copy( Tensor & dst, const Tensor & src ) {
    assert( tn_same_shape( dst, src ) );
    memcpy( dst.data, src.data, sizeof( float ) * src.count );
}

void tn_free( Tensor & tn ) {
    free( tn.data );
    tn = Tensor{};
}

bool tn_same_shape( const Tensor & a, const Tensor & b ) {
    if ( a.dim != b.dim || a.count != b.count ) {
        return false;
    }
    for ( int i = 0; i < a.dim; i++ ) {
        if ( a.shape[i] != b.shape[i] ) {
            return false;
        }
    }
    return true;
}

// ---- Element access --------------------------------------------------------

float & tn_at( Tensor & tn, int i ) {
    assert( i >= 0 && i < tn.count );
    return tn.data[i];
}

float tn_at( const Tensor & tn, int i ) {
    assert( i >= 0 && i < tn.count );
    return tn.data[i];
}

float & tn_at( Tensor & tn, int row, int col ) {
    assert( tn.dim == 2 );
    assert( row >= 0 && row < tn.shape[0] && col >= 0 && col < tn.shape[1] );
    return tn.data[row * tn.shape[1] + col];
}

float tn_at( const Tensor & tn, int row, int col ) {
    assert( tn.dim == 2 );
    assert( row >= 0 && row < tn.shape[0] && col >= 0 && col < tn.shape[1] );
    return tn.data[row * tn.shape[1] + col];
}

// Value of a single-element tensor (e.g. the critic's output).
float tn_value( const Tensor & tn ) {
    assert( tn.count == 1 );
    return tn.data[0];
}

// ---- Fill / init -----------------------------------------------------------

void tn_fill( Tensor & tn, float value ) {
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] = value;
    }
}

void tn_zero( Tensor & tn ) {
    memset( tn.data, 0, sizeof( float ) * tn.count );
}

void tn_seed( unsigned int seed ) {
    tn_rng().seed( seed );
}

void tn_rand_uniform( Tensor & tn, float lo, float hi ) {
    std::uniform_real_distribution<float> dist( lo, hi );
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] = dist( tn_rng() );
    }
}

void tn_rand_normal( Tensor & tn, float mean, float stddev ) {
    std::normal_distribution<float> dist( mean, stddev );
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] = dist( tn_rng() );
    }
}

// ---- In-place arithmetic ---------------------------------------------------

void tn_mul( float scalar, Tensor & tn ) {
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] *= scalar;
    }
}

void tn_add( Tensor & a, const Tensor & b ) {
    assert( a.count == b.count );
    for ( int i = 0; i < a.count; i++ ) {
        a.data[i] += b.data[i];
    }
}

void tn_sub( Tensor & a, const Tensor & b ) {
    assert( a.count == b.count );
    for ( int i = 0; i < a.count; i++ ) {
        a.data[i] -= b.data[i];
    }
}

void tn_hadamard( Tensor & a, const Tensor & b ) {
    assert( a.count == b.count );
    for ( int i = 0; i < a.count; i++ ) {
        a.data[i] *= b.data[i];
    }
}

void tn_axpy( Tensor & a, float scalar, const Tensor & b ) {
    assert( a.count == b.count );
    for ( int i = 0; i < a.count; i++ ) {
        a.data[i] += scalar * b.data[i];
    }
}

void tn_add_scalar( Tensor & tn, float scalar ) {
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] += scalar;
    }
}

void tn_clamp( Tensor & tn, float lo, float hi ) {
    for ( int i = 0; i < tn.count; i++ ) {
        float v = tn.data[i];
        tn.data[i] = v < lo ? lo : ( v > hi ? hi : v );
    }
}

// ---- Reductions ------------------------------------------------------------

float tn_sum( const Tensor & tn ) {
    float s = 0.0f;
    for ( int i = 0; i < tn.count; i++ ) {
        s += tn.data[i];
    }
    return s;
}

float tn_mean( const Tensor & tn ) {
    assert( tn.count > 0 );
    return tn_sum( tn ) / (float)tn.count;
}

float tn_max( const Tensor & tn ) {
    return tn.data[tn_argmax( tn )];
}

int tn_argmax( const Tensor & tn ) {
    assert( tn.count > 0 );
    int best = 0;
    for ( int i = 1; i < tn.count; i++ ) {
        if ( tn.data[i] > tn.data[best] ) {
            best = i;
        }
    }
    return best;
}

float tn_dot( const Tensor & a, const Tensor & b ) {
    assert( a.count == b.count );
    float s = 0.0f;
    for ( int i = 0; i < a.count; i++ ) {
        s += a.data[i] * b.data[i];
    }
    return s;
}

float tn_norm( const Tensor & tn ) {
    return sqrtf( tn_dot( tn, tn ) );
}

// ---- Linear algebra --------------------------------------------------------

void tn_matvec( Tensor & out, const Tensor & m, const Tensor & v ) {
    assert( m.dim == 2 );
    int rows = m.shape[0];
    int cols = m.shape[1];
    assert( v.count == cols );
    assert( out.count == rows );
    assert( out.data != v.data );
    for ( int r = 0; r < rows; r++ ) {
        const float * row = m.data + r * cols;
        float s = 0.0f;
        for ( int c = 0; c < cols; c++ ) {
            s += row[c] * v.data[c];
        }
        out.data[r] = s;
    }
}

// ---- Policy helpers --------------------------------------------------------

void tn_softmax( Tensor & tn ) {
    float m = tn_max( tn );
    float s = 0.0f;
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] = expf( tn.data[i] - m );
        s += tn.data[i];
    }
    for ( int i = 0; i < tn.count; i++ ) {
        tn.data[i] /= s;
    }
}

int tn_sample_categorical( const Tensor & probs ) {
    assert( probs.count > 0 );
    std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
    float u = dist( tn_rng() );
    float cumulative = 0.0f;
    for ( int i = 0; i < probs.count; i++ ) {
        cumulative += probs.data[i];
        if ( u < cumulative ) {
            return i;
        }
    }
    return probs.count - 1; // guard against rounding when probs sum to slightly < 1
}

void tn_one_hot( Tensor & tn, int index ) {
    assert( index >= 0 && index < tn.count );
    tn_zero( tn );
    tn.data[index] = 1.0f;
}

// ---- Debug -----------------------------------------------------------------

void tn_print( const Tensor & tn, const char * name ) {
    if ( name != nullptr ) {
        printf( "%s ", name );
    }
    printf( "[" );
    for ( int i = 0; i < tn.dim; i++ ) {
        printf( i == 0 ? "%d" : "x%d", tn.shape[i] );
    }
    printf( "]\n" );

    if ( tn.dim == 2 ) {
        for ( int r = 0; r < tn.shape[0]; r++ ) {
            printf( "  " );
            for ( int c = 0; c < tn.shape[1]; c++ ) {
                printf( "%9.4f ", tn.data[r * tn.shape[1] + c] );
            }
            printf( "\n" );
        }
    } else {
        printf( "  " );
        for ( int i = 0; i < tn.count; i++ ) {
            printf( "%9.4f ", tn.data[i] );
        }
        printf( "\n" );
    }
}
