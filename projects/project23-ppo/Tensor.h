#pragma once

// Small dense float tensor. Plain struct, manual memory: every tensor returned
// by a tn_create* / tn_clone function owns its buffer and must be released with
// tn_free. Copying the struct (e.g. `Tensor b = a;`) is a shallow copy that
// shares the same buffer.

constexpr int TN_MAX_DIMS = 4;

struct Tensor {
    float * data = nullptr;
    int     count = 0;                  // total number of elements
    int     dim = 0;                    // number of dimensions in use
    int     shape[TN_MAX_DIMS] = {};    // size of each dimension, row-major
};

// ---- Lifetime --------------------------------------------------------------
Tensor  tn_create( int n );                     // 1D, zero-filled
Tensor  tn_create( int rows, int cols );        // 2D, zero-filled
Tensor  tn_create_shape( int dim, const int * shape );
Tensor  tn_clone( const Tensor & tn );          // deep copy
void    tn_copy( Tensor & dst, const Tensor & src ); // copy values, shapes must match
void    tn_free( Tensor & tn );

bool    tn_same_shape( const Tensor & a, const Tensor & b );

// ---- Element access --------------------------------------------------------
float & tn_at( Tensor & tn, int i );
float   tn_at( const Tensor & tn, int i );
float & tn_at( Tensor & tn, int row, int col );
float   tn_at( const Tensor & tn, int row, int col );

// ---- Fill / init -----------------------------------------------------------
void    tn_fill( Tensor & tn, float value );
void    tn_zero( Tensor & tn );
void    tn_seed( unsigned int seed );
void    tn_rand_uniform( Tensor & tn, float lo, float hi );
void    tn_rand_normal( Tensor & tn, float mean, float stddev );

// ---- In-place arithmetic ---------------------------------------------------
void    tn_mul( float scalar, Tensor & tn );                    // tn *= scalar
void    tn_add( Tensor & a, const Tensor & b );                 // a += b
void    tn_sub( Tensor & a, const Tensor & b );                 // a -= b
void    tn_hadamard( Tensor & a, const Tensor & b );            // a *= b (elementwise)
void    tn_axpy( Tensor & a, float scalar, const Tensor & b );  // a += scalar * b
void    tn_add_scalar( Tensor & tn, float scalar );             // tn += scalar
void    tn_clamp( Tensor & tn, float lo, float hi );

// ---- Reductions ------------------------------------------------------------
float   tn_sum( const Tensor & tn );
float   tn_mean( const Tensor & tn );
float   tn_max( const Tensor & tn );
int     tn_argmax( const Tensor & tn );
float   tn_dot( const Tensor & a, const Tensor & b );
float   tn_norm( const Tensor & tn );                           // L2 norm

// ---- Linear algebra --------------------------------------------------------
void    tn_matvec( Tensor & out, const Tensor & m, const Tensor & v );  // out = m * v, m is [rows, cols]

// ---- Policy helpers --------------------------------------------------------
void    tn_softmax( Tensor & tn );                              // in place, numerically stable
int     tn_sample_categorical( const Tensor & probs );          // draw an index from a probability vector
void    tn_one_hot( Tensor & tn, int index );                   // zero tn, set tn[index] = 1

// ---- Debug -----------------------------------------------------------------
void    tn_print( const Tensor & tn, const char * name = nullptr );
