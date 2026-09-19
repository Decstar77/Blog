#pragma once

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

static_assert( sizeof( i8 ) == 1, "Type size mismatch" );
static_assert( sizeof( i16 ) == 2, "Type size mismatch" );
static_assert( sizeof( i32 ) == 4, "Type size mismatch" );
static_assert( sizeof( i64 ) == 8, "Type size mismatch" );
static_assert( sizeof( u8 ) == 1, "Type size mismatch" );
static_assert( sizeof( u16 ) == 2, "Type size mismatch" );
static_assert( sizeof( u32 ) == 4, "Type size mismatch" );
static_assert( sizeof( u64 ) == 8, "Type size mismatch" );
static_assert( sizeof( f32 ) == 4, "Type size mismatch" );
static_assert( sizeof( f64 ) == 8, "Type size mismatch" );

constexpr f32 kPi = 3.14159265358979323846f;
constexpr f32 kTwoPi = 2.0f * kPi;
constexpr f32 kHalfPi = 0.5f * kPi;
constexpr f32 kInvPi = 1.0f / kPi;
constexpr f32 kDeg2Rad = kPi / 180.0f;
constexpr f32 kRad2Deg = 180.0f / kPi;

inline f32 Max( f32 a, f32 b ) { return a > b ? a : b; }
inline f32 Min( f32 a, f32 b ) { return a < b ? a : b; }