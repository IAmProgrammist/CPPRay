#ifndef SHARED_STRUCTS
#define SHARED_STRUCTS

#include <hiprt/hiprt_device.h>
#include <hiprt/hiprt_vec.h>

struct u84
{
	u8 r, g, b, a;
};

struct Texture
{
	u84*		 data;
	int			 size;
};

struct Material {
	int		baseColorIndex;
	int		roughnessIndex;
	int		metalnessIndex;
	int		normalIndex;
	int		emissionIndex;

	Material( int baseColor, int roughness, int metalness, int normal, int emission )
		: baseColorIndex( baseColor ), roughnessIndex( roughness ), metalnessIndex( metalness ), normalIndex( normal ),
		  emissionIndex( emission ){};
};

#endif