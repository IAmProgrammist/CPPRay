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
	Geometry*  geometry,
	Texture*   textures,
	Material*  materials,
	int*	   materialIndices,
	hipLights  lights,
	Camera	   cam,
	int		   frameTime,
	float3* debug) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int	   camType = CAMERA_TYPE_PERSPECTIVE;
	float3 o, d;

	float ar = static_cast<float>( res.x ) / res.y;
	float sh	 = 2 / ( 1 + ar );
	float sw	 = 2 - sh;

	if ( camType == CAMERA_TYPE_PERSPECTIVE ) {
		o = cam.getPosition();
		d = {
			( x / static_cast<float>( res.x ) ) * sw - sw / 2,
			-( ( y / static_cast<float>( res.y ) ) * sh - sh / 2 ),
			( -sw / 2 ) / tanf( degToRad( cam.getFov() / 2 ) ) };
		
		d = cam.getRotatedVector( d );
	} else if ( camType == CAMERA_TYPE_ORTOGRAPHIC ) {
		o = {
			cam.getPosition().x + ( x / static_cast<float>( res.x ) ) * sw - sw / 2,
			cam.getPosition().y - ( ( y / static_cast<float>( res.y ) ) * sh - sh / 2 ),
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
	float cosAngle = 1;
	u84 baseColor;

	if ( !hit.hasHit() ) {
		pixels[pixelIndex * 4 + 0] = 0;
		pixels[pixelIndex * 4 + 1] = 0;
		pixels[pixelIndex * 4 + 2] = 0;
		pixels[pixelIndex * 4 + 3] = 255;

		return;
	}

	hiprtFloat3 N1		  = geometry[hit.instanceID].normals[geometry[hit.instanceID].indices[hit.primID].x];
	hiprtFloat3 N2		  = geometry[hit.instanceID].normals[geometry[hit.instanceID].indices[hit.primID].y];
	hiprtFloat3 N3		  = geometry[hit.instanceID].normals[geometry[hit.instanceID].indices[hit.primID].z];
	float		u		  = hit.uv.x;
	float		v		  = hit.uv.y;
	float		w		  = 1 - hit.uv.x - hit.uv.y;
	hiprtFloat3 hitNormal = normalize(w * N1 + u * N2 + v * N3);
	//baseColor = getAt( hit.uv, textures[materials[materialIndices[hit.instanceID]].baseColorIndex] );
	cosAngle				   = cos( hitNormal, d );
		
	baseColor = { 255, 255, 255, 255 };

	float3 lIntensity = {0.0, 0.0, 0.0};

	float3 currentPoint = o + d * hit.t;
	// Point lights
	for ( int i = 0; i < lights.pointLightsAmount; i++ ) {
		auto  pointLight = lights.pointLights[i];

		hiprtRay lightRay;
		lightRay.origin = pointLight.o;
		lightRay.direction = currentPoint - pointLight.o;
		hiprtSceneTraversalClosest lightTC(
			scene, lightRay, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, ft );
		hiprtHit lightHit = lightTC.getNextHit();

		// Creating a shadow
		if ( lightHit.instanceID != hit.instanceID || lightHit.primID != hit.primID ) continue;

		float distance = len( currentPoint - pointLight.o );
		float attentuation =
			max( min( 1 - ( distance / pointLight.range ) * ( distance / pointLight.range ) * ( distance / pointLight.range ) * (distance / pointLight.range),
					  1 ),
				 0 ) /
			( distance * distance );

		// Dividing by 543.5141306588226 is converting to watts
		lIntensity += pointLight.color * pointLight.intensity * attentuation * cos( -hitNormal, currentPoint - pointLight.o ) /
					  543.5141306588226;
	}

	// Directional lights
	for ( int i = 0; i < lights.dirLightsAmount; i++ ) {
		auto dirLight = lights.dirLights[i];

		hiprtRay lightRay;
		lightRay.origin	   = currentPoint - 0.00001 * dirLight.d;
		lightRay.direction = -dirLight.d;
		
		hiprtSceneTraversalClosest lightTC(
			scene, lightRay, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, ft );
		hiprtHit lightHit = lightTC.getNextHit();

		// Creating a shadow
		if ( lightHit.hasHit() && lightHit.t > 0.001 ) continue;

		// Dividing by 683 is converting to watts
		lIntensity = dirLight.color * dirLight.intensity * cos( -hitNormal, dirLight.d ) / 683;
	}

	// Spot lights
	for ( int i = 0; i < lights.spLightsAmount; i++ ) {
		auto spLight = lights.spLights[i];

		hiprtRay lightRay;
		lightRay.origin	   = spLight.o;
		lightRay.direction = currentPoint - spLight.o;
		hiprtSceneTraversalClosest lightTC(
			scene, lightRay, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, ft );
		hiprtHit lightHit = lightTC.getNextHit();

		// Creating a shadow
		if ( lightHit.instanceID != hit.instanceID || lightHit.primID != hit.primID ) continue;

		float distance	   = len( currentPoint - spLight.o );
		float attentuation = max( min( 1 - ( distance / spLight.range ) * ( distance / spLight.range ) *
											   ( distance / spLight.range ) * ( distance / spLight.range ),
									   1 ),
								  0 ) /
							 ( distance * distance );
		
		float angle = acosf( cos( lightRay.direction, spLight.d ) );

		
		float k;
		if ( angle < spLight.innerConeAngle ) {
			k = 1;
		} else if ( angle > spLight.outerConeAngle ) {
			k = 0;
		} else {
			k = ( angle ) / ( spLight.innerConeAngle - spLight.outerConeAngle ) + 1 -
				( spLight.innerConeAngle ) / ( spLight.innerConeAngle - spLight.outerConeAngle );
		}

		if ( i == 0 ) {
			printf(
				"%f %f %f\n%f %f %f\n%f\n",
				-hitNormal.x,
				-hitNormal.y,
				-hitNormal.z,
				lightRay.direction.x,
				lightRay.direction.y,
				lightRay.direction.z,
				cos( lightRay.direction, -hitNormal ) );
		}
		// Dividing by 543.5141306588226 is converting to watts
		lIntensity += k * spLight.color * spLight.intensity * cos( lightRay.direction, -hitNormal) * attentuation / 54.35141306588226;
	}

	lIntensity = { max( MIN_LIGHT, lIntensity.x ), max( MIN_LIGHT, lIntensity.y ), max( MIN_LIGHT, lIntensity.z ) };

	pixels[pixelIndex * 4 + 0] = min(max( static_cast<unsigned long long>(baseColor.r) * lIntensity.x, 0 ), 255);
	pixels[pixelIndex * 4 + 1] = min(max( static_cast<unsigned long long>(baseColor.g) * lIntensity.y, 0 ), 255);
	pixels[pixelIndex * 4 + 2] = min(max( static_cast<unsigned long long>(baseColor.b) * lIntensity.z, 0 ), 255);
	pixels[pixelIndex * 4 + 3] = max( baseColor.a, 0 );

	//pixels[pixelIndex * 4 + 0] = max( baseColor.r * hitNormal.x, 0 );
	//pixels[pixelIndex * 4 + 1] = max( baseColor.g * hitNormal.y, 0 );
	//pixels[pixelIndex * 4 + 2] = max( baseColor.b * hitNormal.z, 0 );
	//pixels[pixelIndex * 4 + 3] = max( baseColor.a, 0 );
}