//
// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
#include <common/Common.h>
#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 16
#endif

constexpr float Pi	  = 3.14159265358979323846f;
constexpr float TwoPi = 2.0f * Pi;

__device__ u84 getAt( hiprtFloat2& uv, Texture& texture ) {
	// TODO: �������� �������� ������� ������� �� �������� ����������.
	hiprtInt2 rootIndex = {
		( (int)( uv.x * texture.size ) ) % texture.size, ( (int)( ( 1 - uv.y ) * texture.size ) ) % texture.size };
	return texture.data[rootIndex.x + rootIndex.y * texture.size];
}

extern "C" __global__ void SceneIntersectionKernel(
	hiprtScene scene,
	u8*		   pixels,
	int2	   res,
	Texture*   textures,
	Material*  materials,
	int*	   materialIndices,
	Camera	   cam,
	int		   frameTime ) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int	   camType = CAMERA_TYPE_PERSPECTIVE;
	float3 o, d;

	float ar = static_cast<float>( res.x ) / res.y;
	float h	 = 2 / ( 1 + ar );
	float w	 = 2 - h;

	if ( camType == CAMERA_TYPE_PERSPECTIVE ) {
		o = cam.getPosition();
		d = {
			( x / static_cast<float>( res.x ) ) * w - w / 2,
			-( ( y / static_cast<float>( res.y ) ) * h - h / 2 ),
			( -w / 2 ) / tanf( degToRad( cam.getFov() / 2 ) ) };
		auto phi = cam.getRotation().w;
		auto k	 = make_hiprtFloat3( cam.getRotation().x, cam.getRotation().y, cam.getRotation().z );
		d		 = d * cos( phi ) + cross( k, d ) * sin( phi ) + k * ( k * d ) * ( 1 - cos( phi ) );
	} else if ( camType == CAMERA_TYPE_ORTOGRAPHIC ) {
		o = {
			cam.getPosition().x + ( x / static_cast<float>( res.x ) ) * w - w / 2,
			cam.getPosition().y - ( ( y / static_cast<float>( res.y ) ) * h - h / 2 ),
			cam.getPosition().z };
		d = { 0.0f, 0.0f, -1 };
		// TODO: �������� ���������� �������
		//auto phi = cam.getRotation().w;
		//auto k	 = make_hiprtFloat3( cam.getRotation().x, cam.getRotation().y, cam.getRotation().z );
		//d		 = d * cos( phi ) + cross( k, d ) * sin( phi ) + k * ( k * d ) * ( 1 - cos( phi ) );
	}

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	float					   ft = frameTime;
	hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, ft );
	hiprtHit				   hit = tr.getNextHit();

	int pixelIndex = x + y * res.x;
	u84 baseColor;
	if ( hit.hasHit() ) {
		//printf( "VECTORY!!!\n" );
		//printf( "%f %f %f %f %f %f\n", hit.normal.x, hit.normal.y, hit.normal.z,
		//	ray.direction.x, ray.direction.y, ray.direction.z);
		baseColor = getAt( hit.uv, textures[materials[materialIndices[hit.instanceID]].baseColorIndex] );
		//printf( "%d %d %d %f\n", baseColor.r, baseColor.g, baseColor.b, cos( hit.normal, ray.direction ) );
	}  else
		baseColor = { 0, 0, 0, 0 };
	float cosAngle = ( cos( hit.normal, ray.direction ) );

	pixels[pixelIndex * 4 + 0] = max( baseColor.r * cosAngle, 0 );
	pixels[pixelIndex * 4 + 1] = max( baseColor.g * cosAngle, 0 );
	pixels[pixelIndex * 4 + 2] = max( baseColor.b * cosAngle, 0 );
	pixels[pixelIndex * 4 + 3] = max( baseColor.a * cosAngle, 0 );
}