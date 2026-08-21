#pragma once
namespace nerf {
    using i8 = char;
    using i16 = short;
    using i32 = int;
    using i64 = long long;

    using u8 = unsigned char;
    using u16 = unsigned short;
    using u32 = unsigned int;
    using u64 = unsigned long long;

    using f32 = float;
    using f64 = double;

    static_assert( sizeof( i8 ) == 1 );
    static_assert( sizeof( i16 ) == 2 );
    static_assert( sizeof( i32 ) == 4 );
    static_assert( sizeof( i64 ) == 8 );
    static_assert( sizeof( u8 ) == 1 );
    static_assert( sizeof( u16 ) == 2 );
    static_assert( sizeof( u32 ) == 4 );
    static_assert( sizeof( u64 ) == 8 );
    static_assert( sizeof( f32 ) == 4 );
    static_assert( sizeof( f64 ) == 8 );

    constexpr f32 kPi = 3.14159265358979323846f;
    constexpr f32 kTwoPi = 2.0f * kPi;
    constexpr f32 kHalfPi = 0.5f * kPi;
    constexpr f32 kInvPi = 1.0f / kPi;
    constexpr f32 kDeg2Rad = kPi / 180.0f;
    constexpr f32 kRad2Deg = 180.0f / kPi;

    struct StringBuffer {
        char *  data;
        i32     count;
        i32     cap;
    };

    struct LargeString {
        char    data[256];
        i32     count;
    };

    struct SmallString {
        char    data[64];
        i32     count;
    };
} // namespace nerf

#define SPLATS_ARRAY_COUNT( arr ) ( sizeof( arr ) / sizeof( ( arr )[0] ) )
#define SPLATS_UNUSED( x ) ( (void) ( x ) )

#define SPLATS_KB( n ) ( ( n ) * 1024ull )
#define SPLATS_MB( n ) ( ( n ) * 1024ull * 1024ull )
#define SPLATS_GB( n ) ( ( n ) * 1024ull * 1024ull * 1024ull )
