#pragma once

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#if !defined( __KERNELCC__ )
#include <cmath>
#endif

#if !defined( __KERNELCC__ )
#define HOST
#define DEVICE
#define HOST_DEVICE

#define int2 hiprtInt2
#define int3 hiprtInt3
#define int4 hiprtInt4

#define float2 hiprtFloat2
#define float3 hiprtFloat3
#define float4 hiprtFloat4

#define make_int2 make_hiprtInt2
#define make_int3 make_hiprtInt3
#define make_int4 make_hiprtInt4

#define make_float2 make_hiprtFloat2
#define make_float3 make_hiprtFloat3
#define make_float4 make_hiprtFloat4

#else
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#endif

#ifdef __CUDACC__
#define INLINE __forceinline__
#else
#define INLINE inline
#endif

#define CAMERA_TYPE_PERSPECTIVE 0
#define CAMERA_TYPE_ORTOGRAPHIC 1

#define RT_MIN( a, b ) ( ( ( b ) < ( a ) ) ? ( b ) : ( a ) )
#define RT_MAX( a, b ) ( ( ( b ) > ( a ) ) ? ( b ) : ( a ) )

HOST_DEVICE INLINE int2 make_int2( const float2 a ) { return make_int2( (int)a.x, (int)a.y ); }

HOST_DEVICE INLINE int2 make_int2( const int3& a ) { return make_int2( a.x, a.y ); }

HOST_DEVICE INLINE int2 make_int2( const int4& a ) { return make_int2( a.x, a.y ); }

HOST_DEVICE INLINE int2 make_int2( const int c ) { return make_int2( c, c ); }

HOST_DEVICE INLINE int2 operator+( const int2& a, const int2& b ) { return make_int2( a.x + b.x, a.y + b.y ); }

HOST_DEVICE INLINE int2 operator-( const int2& a, const int2& b ) { return make_int2( a.x - b.x, a.y - b.y ); }

HOST_DEVICE INLINE int2 operator*( const int2& a, const int2& b ) { return make_int2( a.x * b.x, a.y * b.y ); }

HOST_DEVICE INLINE int2 operator/( const int2& a, const int2& b ) { return make_int2( a.x / b.x, a.y / b.y ); }

HOST_DEVICE INLINE int2& operator+=( int2& a, const int2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator-=( int2& a, const int2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator*=( int2& a, const int2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator/=( int2& a, const int2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HOST_DEVICE INLINE int2& operator+=( int2& a, const int c )
{
	a.x += c;
	a.y += c;
	return a;
}

HOST_DEVICE INLINE int2& operator-=( int2& a, const int c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HOST_DEVICE INLINE int2& operator*=( int2& a, const int c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HOST_DEVICE INLINE int2& operator/=( int2& a, const int c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HOST_DEVICE INLINE int2 operator-( const int2& a ) { return make_int2( -a.x, -a.y ); }

HOST_DEVICE INLINE int2 operator+( const int2& a, const int c ) { return make_int2( a.x + c, a.y + c ); }

HOST_DEVICE INLINE int2 operator+( const int c, const int2& a ) { return make_int2( c + a.x, c + a.y ); }

HOST_DEVICE INLINE int2 operator-( const int2& a, const int c ) { return make_int2( a.x - c, a.y - c ); }

HOST_DEVICE INLINE int2 operator-( const int c, const int2& a ) { return make_int2( c - a.x, c - a.y ); }

HOST_DEVICE INLINE int2 operator*( const int2& a, const int c ) { return make_int2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE int2 operator*( const int c, const int2& a ) { return make_int2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE int2 operator/( const int2& a, const int c ) { return make_int2( a.x / c, a.y / c ); }

HOST_DEVICE INLINE int2 operator/( const int c, const int2& a ) { return make_int2( c / a.x, c / a.y ); }

HOST_DEVICE INLINE int3 make_int3( const float3& a ) { return make_int3( (int)a.x, (int)a.y, (int)a.z ); }

HOST_DEVICE INLINE int3 make_int3( const int4& a ) { return make_int3( a.x, a.y, a.z ); }

HOST_DEVICE INLINE int3 make_int3( const int2& a, const int c ) { return make_int3( a.x, a.y, c ); }

HOST_DEVICE INLINE int3 make_int3( const int c ) { return make_int3( c, c, c ); }

HOST_DEVICE INLINE int3 operator+( const int3& a, const int3& b ) { return make_int3( a.x + b.x, a.y + b.y, a.z + b.z ); }

HOST_DEVICE INLINE int3 operator-( const int3& a, const int3& b ) { return make_int3( a.x - b.x, a.y - b.y, a.z - b.z ); }

HOST_DEVICE INLINE int3 operator*( const int3& a, const int3& b ) { return make_int3( a.x * b.x, a.y * b.y, a.z * b.z ); }

HOST_DEVICE INLINE int3 operator/( const int3& a, const int3& b ) { return make_int3( a.x / b.x, a.y / b.y, a.z / b.z ); }

HOST_DEVICE INLINE int3& operator+=( int3& a, const int3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator-=( int3& a, const int3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator*=( int3& a, const int3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator/=( int3& a, const int3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HOST_DEVICE INLINE int3& operator+=( int3& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HOST_DEVICE INLINE int3& operator-=( int3& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HOST_DEVICE INLINE int3& operator*=( int3& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HOST_DEVICE INLINE int3& operator/=( int3& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HOST_DEVICE INLINE int3 operator-( const int3& a ) { return make_int3( -a.x, -a.y, -a.z ); }

HOST_DEVICE INLINE int3 operator+( const int3& a, const int c ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HOST_DEVICE INLINE int3 operator+( const int c, const int3& a ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HOST_DEVICE INLINE int3 operator-( const int3& a, const int c ) { return make_int3( a.x - c, a.y - c, a.z - c ); }

HOST_DEVICE INLINE int3 operator-( const int c, const int3& a ) { return make_int3( c - a.x, c - a.y, c - a.z ); }

HOST_DEVICE INLINE int3 operator*( const int3& a, const int c ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HOST_DEVICE INLINE int3 operator*( const int c, const int3& a ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HOST_DEVICE INLINE int3 operator/( const int3& a, const int c ) { return make_int3( a.x / c, a.y / c, a.z / c ); }

HOST_DEVICE INLINE int3 operator/( const int c, const int3& a ) { return make_int3( c / a.x, c / a.y, c / a.z ); }

HOST_DEVICE INLINE int4 make_int4( const float4& a ) { return make_int4( (int)a.x, (int)a.y, (int)a.z, (int)a.w ); }

HOST_DEVICE INLINE int4 make_int4( const int2& a, const int c0, const int c1 ) { return make_int4( a.x, a.y, c0, c1 ); }

HOST_DEVICE INLINE int4 make_int4( const int3& a, const int c ) { return make_int4( a.x, a.y, a.z, c ); }

HOST_DEVICE INLINE int4 make_int4( const int c ) { return make_int4( c, c, c, c ); }

HOST_DEVICE INLINE int4 operator+( const int4& a, const int4& b )
{
	return make_int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HOST_DEVICE INLINE int4 operator-( const int4& a, const int4& b )
{
	return make_int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HOST_DEVICE INLINE int4 operator*( const int4& a, const int4& b )
{
	return make_int4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HOST_DEVICE INLINE int4 operator/( const int4& a, const int4& b )
{
	return make_int4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

HOST_DEVICE INLINE int4& operator+=( int4& a, const int4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator-=( int4& a, const int4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator*=( int4& a, const int4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator/=( int4& a, const int4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HOST_DEVICE INLINE int4& operator+=( int4& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HOST_DEVICE INLINE int4& operator-=( int4& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HOST_DEVICE INLINE int4& operator*=( int4& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HOST_DEVICE INLINE int4& operator/=( int4& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HOST_DEVICE INLINE int4 operator-( const int4& a ) { return make_int4( -a.x, -a.y, -a.z, -a.w ); }

HOST_DEVICE INLINE int4 operator+( const int4& a, const int c ) { return make_int4( c + a.x, c + a.y, c + a.z, c + a.w ); }

HOST_DEVICE INLINE int4 operator+( const int c, const int4& a ) { return make_int4( c + a.x, c + a.y, c + a.z, c + a.w ); }

HOST_DEVICE INLINE int4 operator-( const int4& a, const int c ) { return make_int4( a.x - c, a.y - c, a.z - c, a.w - c ); }

HOST_DEVICE INLINE int4 operator-( const int c, const int4& a ) { return make_int4( c - a.x, c - a.y, c - a.z, c - a.w ); }

HOST_DEVICE INLINE int4 operator*( const int4& a, const int c ) { return make_int4( c * a.x, c * a.y, c * a.z, c * a.w ); }

HOST_DEVICE INLINE int4 operator*( const int c, const int4& a ) { return make_int4( c * a.x, c * a.y, c * a.z, c * a.w ); }

HOST_DEVICE INLINE int4 operator/( const int4& a, const int c ) { return make_int4( a.x / c, a.y / c, a.z / c, a.w / c ); }

HOST_DEVICE INLINE int4 operator/( const int c, const int4& a ) { return make_int4( c / a.x, c / a.y, c / a.z, c / a.w ); }

HOST_DEVICE INLINE int2 max( const int2& a, const int2& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 max( const int2& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 max( const int c, const int2& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 min( const int2& a, const int2& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 min( const int2& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int2 min( const int c, const int2& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	return make_int2( x, y );
}

HOST_DEVICE INLINE int3 max( const int3& a, const int3& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	int z = RT_MAX( a.z, b.z );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 max( const int3& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 max( const int c, const int3& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 min( const int3& a, const int3& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	int z = RT_MIN( a.z, b.z );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 min( const int3& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int3 min( const int c, const int3& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	return make_int3( x, y, z );
}

HOST_DEVICE INLINE int4 max( const int4& a, const int4& b )
{
	int x = RT_MAX( a.x, b.x );
	int y = RT_MAX( a.y, b.y );
	int z = RT_MAX( a.z, b.z );
	int w = RT_MAX( a.w, b.w );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 max( const int4& a, const int c )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	int w = RT_MAX( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 max( const int c, const int4& a )
{
	int x = RT_MAX( a.x, c );
	int y = RT_MAX( a.y, c );
	int z = RT_MAX( a.z, c );
	int w = RT_MAX( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 min( const int4& a, const int4& b )
{
	int x = RT_MIN( a.x, b.x );
	int y = RT_MIN( a.y, b.y );
	int z = RT_MIN( a.z, b.z );
	int w = RT_MIN( a.w, b.w );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 min( const int4& a, const int c )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	int w = RT_MIN( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE int4 min( const int c, const int4& a )
{
	int x = RT_MIN( a.x, c );
	int y = RT_MIN( a.y, c );
	int z = RT_MIN( a.z, c );
	int w = RT_MIN( a.w, c );
	return make_int4( x, y, z, w );
}

HOST_DEVICE INLINE float2 make_float2( const int2& a ) { return make_float2( (float)a.x, (float)a.y ); }

HOST_DEVICE INLINE float2 make_float2( const float3& a ) { return make_float2( a.x, a.y ); }

HOST_DEVICE INLINE float2 make_float2( const float4& a ) { return make_float2( a.x, a.y ); }

HOST_DEVICE INLINE float2 make_float2( const float c ) { return make_float2( c, c ); }

HOST_DEVICE INLINE float2 operator+( const float2& a, const float2& b ) { return make_float2( a.x + b.x, a.y + b.y ); }

HOST_DEVICE INLINE float2 operator-( const float2& a, const float2& b ) { return make_float2( a.x - b.x, a.y - b.y ); }

HOST_DEVICE INLINE float2 operator*( const float2& a, const float2& b ) { return make_float2( a.x * b.x, a.y * b.y ); }

HOST_DEVICE INLINE float2 operator/( const float2& a, const float2& b ) { return make_float2( a.x / b.x, a.y / b.y ); }

HOST_DEVICE INLINE float2& operator+=( float2& a, const float2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator-=( float2& a, const float2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator*=( float2& a, const float2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator/=( float2& a, const float2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HOST_DEVICE INLINE float2& operator+=( float2& a, const float c )
{
	a.x += c;
	a.y += c;
	return a;
}

HOST_DEVICE INLINE float2& operator-=( float2& a, const float c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HOST_DEVICE INLINE float2& operator*=( float2& a, const float c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HOST_DEVICE INLINE float2& operator/=( float2& a, const float c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HOST_DEVICE INLINE float2 operator-( const float2& a ) { return make_float2( -a.x, -a.y ); }

HOST_DEVICE INLINE float2 operator+( const float2& a, const float c ) { return make_float2( a.x + c, a.y + c ); }

HOST_DEVICE INLINE float2 operator+( const float c, const float2& a ) { return make_float2( c + a.x, c + a.y ); }

HOST_DEVICE INLINE float2 operator-( const float2& a, const float c ) { return make_float2( a.x - c, a.y - c ); }

HOST_DEVICE INLINE float2 operator-( const float c, const float2& a ) { return make_float2( c - a.x, c - a.y ); }

HOST_DEVICE INLINE float2 operator*( const float2& a, const float c ) { return make_float2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE float2 operator*( const float c, const float2& a ) { return make_float2( c * a.x, c * a.y ); }

HOST_DEVICE INLINE float2 operator/( const float2& a, const float c ) { return make_float2( a.x / c, a.y / c ); }

HOST_DEVICE INLINE float2 operator/( const float c, const float2& a ) { return make_float2( c / a.x, c / a.y ); }

HOST_DEVICE INLINE float3 make_float3( const int3& a ) { return make_float3( (float)a.x, (float)a.y, (float)a.z ); }

HOST_DEVICE INLINE float3 make_float3( const float4& a ) { return make_float3( a.x, a.y, a.z ); }

HOST_DEVICE INLINE float3 make_float3( const float2& a, const float c ) { return make_float3( a.x, a.y, c ); }

HOST_DEVICE INLINE float3 make_float3( const float c ) { return make_float3( c, c, c ); }

HOST_DEVICE INLINE float3 operator+( const float3& a, const float3& b )
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HOST_DEVICE INLINE float3 operator-( const float3& a, const float3& b )
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HOST_DEVICE INLINE float3 operator*( const float3& a, const float3& b )
{
	return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HOST_DEVICE INLINE float3 operator/( const float3& a, const float3& b )
{
	return make_float3( a.x / b.x, a.y / b.y, a.z / b.z );
}

HOST_DEVICE INLINE float3& operator+=( float3& a, const float3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator-=( float3& a, const float3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator*=( float3& a, const float3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator/=( float3& a, const float3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HOST_DEVICE INLINE float3& operator+=( float3& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HOST_DEVICE INLINE float3& operator-=( float3& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HOST_DEVICE INLINE float3& operator*=( float3& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HOST_DEVICE INLINE float3& operator/=( float3& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HOST_DEVICE INLINE float3 operator-( const float3& a ) { return make_float3( -a.x, -a.y, -a.z ); }

HOST_DEVICE INLINE float3 operator+( const float3& a, const float c ) { return make_float3( c + a.x, c + a.y, c + a.z ); }

HOST_DEVICE INLINE float3 operator+( const float c, const float3& a ) { return make_float3( c + a.x, c + a.y, c + a.z ); }

HOST_DEVICE INLINE float3 operator-( const float3& a, const float c ) { return make_float3( a.x - c, a.y - c, a.z - c ); }

HOST_DEVICE INLINE float3 operator-( const float c, const float3& a ) { return make_float3( c - a.x, c - a.y, c - a.z ); }

HOST_DEVICE INLINE float3 operator*( const float3& a, const float c ) { return make_float3( c * a.x, c * a.y, c * a.z ); }

HOST_DEVICE INLINE float3 operator*( const float c, const float3& a ) { return make_float3( c * a.x, c * a.y, c * a.z ); }

HOST_DEVICE INLINE float3 operator/( const float3& a, const float c ) { return make_float3( a.x / c, a.y / c, a.z / c ); }

HOST_DEVICE INLINE float3 operator/( const float c, const float3& a ) { return make_float3( c / a.x, c / a.y, c / a.z ); }

HOST_DEVICE INLINE float4 make_float4( const int4& a ) { return make_float4( (float)a.x, (float)a.y, (float)a.z, (float)a.w ); }

HOST_DEVICE INLINE float4 make_float4( const float2& a, const float c0, const float c1 )
{
	return make_float4( a.x, a.y, c0, c1 );
}

HOST_DEVICE INLINE float4 make_float4( const float3& a, const float c ) { return make_float4( a.x, a.y, a.z, c ); }

HOST_DEVICE INLINE float4 make_float4( const float c ) { return make_float4( c, c, c, c ); }

HOST_DEVICE INLINE float4 operator+( const float4& a, const float4& b )
{
	return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HOST_DEVICE INLINE float4 operator-( const float4& a, const float4& b )
{
	return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HOST_DEVICE INLINE float4 operator*( const float4& a, const float4& b )
{
	return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HOST_DEVICE INLINE float4 operator/( const float4& a, const float4& b )
{
	return make_float4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
}

HOST_DEVICE INLINE float4& operator+=( float4& a, const float4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator-=( float4& a, const float4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator*=( float4& a, const float4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator/=( float4& a, const float4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HOST_DEVICE INLINE float4& operator+=( float4& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HOST_DEVICE INLINE float4& operator-=( float4& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HOST_DEVICE INLINE float4& operator*=( float4& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HOST_DEVICE INLINE float4& operator/=( float4& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HOST_DEVICE INLINE float4 operator-( const float4& a ) { return make_float4( -a.x, -a.y, -a.z, -a.w ); }

HOST_DEVICE INLINE float4 operator+( const float4& a, const float c )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HOST_DEVICE INLINE float4 operator+( const float c, const float4& a )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HOST_DEVICE INLINE float4 operator-( const float4& a, const float c )
{
	return make_float4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HOST_DEVICE INLINE float4 operator-( const float c, const float4& a )
{
	return make_float4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HOST_DEVICE INLINE float4 operator*( const float4& a, const float c )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HOST_DEVICE INLINE float4 operator*( const float c, const float4& a )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HOST_DEVICE INLINE float4 operator/( const float4& a, const float c )
{
	return make_float4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HOST_DEVICE INLINE float4 operator/( const float c, const float4& a )
{
	return make_float4( c / a.x, c / a.y, c / a.z, c / a.w );
}

HOST_DEVICE INLINE float3 cross( const float3& a, const float3& b )
{
	return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

HOST_DEVICE INLINE float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

HOST_DEVICE INLINE float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

HOST_DEVICE INLINE float len( const float3& a ) { return sqrtf( a.x * a.x + a.y * a.y + a.z * a.z ); };

HOST_DEVICE INLINE float cos( const float3& a, const float3& b ) { return dot( a, b ) / ( len( a ) * len(b)); }

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#define PI 3.14159265358979323846f

HOST_DEVICE INLINE float degToRad( const float& deg ) { return ( deg * PI ) / 180.0f; }
HOST_DEVICE INLINE float radToDeg( const float& rad ) { return ( rad * 180.0f ) / PI; }

struct u84 {
	u8 r, g, b, a;
};

struct Texture {
	u84* data;
	int	 size;
};

struct Material {
	int baseColorIndex;
	int roughnessIndex;
	int metalnessIndex;
	int normalIndex;
	int emissionIndex;

	Material( int baseColor, int roughness, int metalness, int normal, int emission )
		: baseColorIndex( baseColor ), roughnessIndex( roughness ), metalnessIndex( metalness ), normalIndex( normal ),
		  emissionIndex( emission ){};
};

struct Camera {
  public:
	// X Y Z
	float3 position = make_float3(0, 0, 0);
	// Euler XYZ
	float4 rotation = make_float4( 1, 0, 0, 0 );

	Camera(float3 position, float4 rotation) { 
		this->position.x = position.x;
		this->position.y = position.y;
		this->position.z = position.z;
		this->rotation.x = rotation.x;
		this->rotation.y = rotation.y;
		this->rotation.z = rotation.z;
		this->rotation.w = rotation.w;
	}

	// TODO: remake
	HOST_DEVICE static float4 getRotationInAxisAngle( float3 rot ) {
		// Assuming the angles are in radians.
		double c1	= cos( rot.z / 2 );
		double s1	= sin( rot.z / 2 );
		double c2	= cos( rot.x / 2 );
		double s2	= sin( rot.x / 2 );
		double c3	= cos( rot.y / 2 );
		double s3	= sin( rot.y / 2 );
		double c1c2 = c1 * c2;
		double s1s2 = s1 * s2;
		float4 answer;
		answer.w			= c1c2 * c3 - s1s2 * s3;
		answer.x			= c1c2 * s3 + s1s2 * c3;
		answer.y			= s1 * c2 * c3 + c1 * s2 * s3;
		answer.z			= c1 * s2 * c3 - s1 * c2 * s3;
		answer.w			= radToDeg(2 * acos( answer.w ));
		double norm = answer.x * answer.x + answer.y * answer.y + answer.z * answer.z;
		if ( norm < 0.001 ) { // when all euler angles are zero angle =0 so
			// we can set axis to anything to avoid divide by zero
			answer.x = 1;
			answer.y = answer.z = 0;
		} else {
			norm = sqrt( norm );
			answer.x /= norm;
			answer.y /= norm;
			answer.z /= norm;
		}

		return answer;
	}

	//HOST_DEVICE float4 getRotationInAxisAngle() { return Camera::getRotationInAxisAngle( this->rotation );
	//}

#ifndef __KERNELCC__
	// Creates frame properties from absolute camera position in world
	// position: x, y, z
	// Rotation: x, y, z, w (angle in deg)
	HOST hiprtFrameSRT fromCameraPos() {
		hiprtFrameSRT frame;
		frame.scale						= make_hiprtFloat3( 1, 1, 1 );
		frame.rotation	  = make_hiprtFloat4( rotation.x, rotation.y, rotation.z, degToRad( -rotation.w ) );
		auto k			  = make_float3( frame.rotation.x, frame.rotation.y, frame.rotation.z );
		auto v			  = make_float3( -position.x, -position.y, -position.z );
		auto phi		  = frame.rotation.w;
		frame.translation = v * cosf( phi ) + ( cross( k, v ) ) * sinf( phi ) + k * ( k * v ) * ( 1 - cosf( phi ) );

		return frame;
	}
#endif
};