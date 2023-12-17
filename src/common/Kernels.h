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

// GGX/Trowbridge-Reitz Normal Distribution Function
__device__ inline float D(float& alpha, float3& N, float3 &H) { 
	float numerator = alpha * alpha;

	float NdotH = max( dot( N, H ), 0 );
	float denominator = Pi * powf( powf( NdotH, 2 ) * ( powf( alpha, 2 ) - 1 ) + 1, 2 );
	denominator		  = max( denominator, 0.000001 );

	return numerator / denominator;
}

__device__ inline float G1( float &alpha, float3 &N, float3& X ) { 
	float numerator = max( dot( N, X ), 0 );

	float k = alpha / 2;
	float denominator = numerator * ( 1.0 - k ) + k;
	denominator		  = max( denominator, 0.000001 );

	return numerator / denominator;
}

__device__ float inline G( float& alpha, float3& N, float3& V, float3& L ) { 
	return G1( alpha, N, V ) * G1( alpha, N, L ); 
}

__device__ inline float3 F( float3& F0, float3& V, float3& H ) { return F0 + ( 1 - F0 ) * powf( 1 - max( dot( V, H ), 0 ), 5 ); }

__device__ inline float3 PBR(float3& F0, float3& N, float3& V, float3& H, float3& L, float3& albedoMesh, Material& material, float& alpha) {
	float3 Ks = F( F0, V, H );
	float3 Kd = ( 1 - Ks ) * ( 1 - material.metallic );

	float3 lambert = albedoMesh / Pi;

	float3 cookTorranceNumerator   = D( alpha, N, H ) * G( alpha, N, V, L ) * Ks;
	float  cookTorranceDenominator = 4 * max( dot( V, N ), 0 ) * max( dot( L, N ), 0 );
	cookTorranceDenominator		   = max( cookTorranceDenominator, 0.000001 );
	float3 cookTorrance			   = cookTorranceNumerator / cookTorranceDenominator;

	float3 BRDF = Kd * lambert + cookTorrance;

	return BRDF;
}

extern "C" __global__ void mainKernel(
	hiprtScene scene,
	u8*		   pixels,
	int2	   res,
	Geometry*  geometry,
	Material*  materials,
	hipLights  lights,
	Camera	   cam) {
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
		// TODO: implement orographic camera
		o = {
			cam.getPosition().x + ( x / static_cast<float>( res.x ) ) * sw - sw / 2,
			cam.getPosition().y - ( ( y / static_cast<float>( res.y ) ) * sh - sh / 2 ),
			cam.getPosition().z };
		d = { 0.0f, 0.0f, -1 };
	}

	hiprtRay ray;
	ray.origin	  = o;
	ray.direction = d;

	hiprtSceneTraversalClosest tr( scene, ray );
	hiprtHit				   hit = tr.getNextHit();

	int pixelIndex = x + y * res.x;
	float cosAngle = 1;
	u84	  baseColor	 = {255, 255, 255, 255};

	if ( !hit.hasHit() ) {
		u84 bg					   = BACKGROUND_COLOR;
		pixels[pixelIndex * 4 + 0] = bg.r;
		pixels[pixelIndex * 4 + 1] = bg.g;
		pixels[pixelIndex * 4 + 2] = bg.b;
		pixels[pixelIndex * 4 + 3] = bg.a;

		return;
	}

	hiprtFloat3 N1		  = geometry[hit.instanceID].normals[geometry[hit.instanceID].indices[hit.primID].x];
	hiprtFloat3 N2		  = geometry[hit.instanceID].normals[geometry[hit.instanceID].indices[hit.primID].y];
	hiprtFloat3 N3		  = geometry[hit.instanceID].normals[geometry[hit.instanceID].indices[hit.primID].z];
	float		u		  = hit.uv.x;
	float		v		  = hit.uv.y;
	float		w		  = 1 - hit.uv.x - hit.uv.y;
	hiprtFloat3 normal = normalize(w * N1 + u * N2 + v * N3);
	auto material         = materials[hit.instanceID];
	float3		albedoMesh	   = { material.baseColorR, material.baseColorG, material.baseColorB };
	cosAngle				   = cos( normal, d );

	hiprtFloat3 lIntensity		 = { MIN_LIGHT, MIN_LIGHT, MIN_LIGHT };
	float3		fragmentPosition	   = hit.t * ray.direction + ray.origin;
	float3		cameraPosition	 = cam.getPosition();

	float3 F0	 = { 0.5, 0.5, 0.5 };
	float3 N = normalize( normal );
	float3 V = normalize( cameraPosition - fragmentPosition );
	float  alpha = material.roughness * material.roughness;

	// Direct lighting

	// Point lights
	for ( int i = 0; i < lights.pointLightsAmount; i++ ) {
		auto pointLight = lights.pointLights[i];

		hiprtRay lightRay;
		lightRay.origin	   = pointLight.o;
		lightRay.direction = fragmentPosition - pointLight.o;
		hiprtSceneTraversalClosest lightTC(scene, lightRay );
		hiprtHit lightHit = lightTC.getNextHit();

		// Creating a shadow
		if ( lightHit.instanceID != hit.instanceID || lightHit.primID != hit.primID ) continue;

		float distance	   = len( fragmentPosition - pointLight.o );
		float attentuation = max( min( 1 - ( distance / pointLight.range ) * ( distance / pointLight.range ) *
											   ( distance / pointLight.range ) * ( distance / pointLight.range ),
									   1 ),
								  0 ) /
							 ( distance * distance );

		float3 L = normalize( lightRay.direction );
		float3 H = normalize( V + L );

		
		lIntensity += PBR( F0, N, V, H, L, albedoMesh, material, alpha ) * ( pointLight.color * pointLight.intensity * attentuation ) *
					  BRIGHTNESS *
					  cos( -normal, fragmentPosition - pointLight.o );

		//lIntensity += pointLight.color * pointLight.intensity * BRIGHTNESS * attentuation *
		//			  cos( -normal, fragmentPosition - pointLight.o );
	}

	// Directional lights
	for ( int i = 0; i < lights.dirLightsAmount; i++ ) {
		auto dirLight = lights.dirLights[i];

		hiprtRay lightRay;
		lightRay.origin	   = fragmentPosition - dirLight.d * 0.0001;
		lightRay.direction = -dirLight.d;

		hiprtSceneTraversalClosest lightTC(	scene, lightRay );
		hiprtHit lightHit = lightTC.getNextHit();

		// Creating a shadow
		if ( lightHit.hasHit() ) continue;

		float3 L = normalize( lightRay.direction );
		float3 H = normalize( V + L );

		lIntensity += PBR( F0, N, V, H, L, albedoMesh, material, alpha ) * dirLight.color * dirLight.intensity * BRIGHTNESS *
					  cos( -normal, dirLight.d );
	}

	// Spot lights
	for ( int i = 0; i < lights.spLightsAmount; i++ ) {
		auto spLight = lights.spLights[i];

		hiprtRay lightRay;
		lightRay.origin	   = spLight.o;
		lightRay.direction = fragmentPosition - spLight.o;
		hiprtSceneTraversalClosest lightTC(	scene, lightRay  );
		hiprtHit lightHit = lightTC.getNextHit();

		// Creating a shadow
		if ( lightHit.instanceID != hit.instanceID || lightHit.primID != hit.primID ) continue;

		float distance	   = len( fragmentPosition - spLight.o );
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

		float3 L = normalize( lightRay.direction );
		float3 H = normalize( V + L );

		lIntensity += PBR( F0, N, V, H, L, albedoMesh, material, alpha ) * k * spLight.color * spLight.intensity * BRIGHTNESS *
					  cos( lightRay.direction, -normal ) * attentuation;
	}

	pixels[pixelIndex * 4 + 0] = min(max( lIntensity.x * 255, 0 ), 255);
	pixels[pixelIndex * 4 + 1] = min( max( lIntensity.y * 255, 0 ), 255 );
	pixels[pixelIndex * 4 + 2] = min( max( lIntensity.z * 255, 0 ), 255 );
	pixels[pixelIndex * 4 + 3] = 255;
}