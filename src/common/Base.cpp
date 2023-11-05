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

#include "Base.h"

#include "Common.h"
#include <assert.h>
#include <tiny_gltf.h>
#include <Eigen/Geometry>

#define MANUAL 0

void checkOro( oroError res, const char* file, int line ) {
	if ( res != oroSuccess ) {
		const char* msg;
		oroGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkOrortc( orortcResult res, const char* file, int line ) {
	if ( res != ORORTC_SUCCESS ) {
		std::cerr << "ORORTC error: '" << orortcGetErrorString( res ) << "' [ " << res << " ] on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkHiprt( hiprtError res, const char* file, int line ) {
	if ( res != hiprtSuccess ) {
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

hiprtFrameMatrix inline operator*( hiprtFrameMatrix& a, hiprtFrameMatrix& b ) { 
	Eigen::Matrix<float, 4, 4> aMat; 
	Eigen::Matrix<float, 4, 4> bMat;
	aMat.setIdentity();
	bMat.setIdentity();
	for ( int x = 0; x < 4; x++ ) {
		for ( int y = 0; y < 4; y++ ) {
			aMat( x, y ) = a.matrix[x][y];
			bMat( x, y ) = b.matrix[x][y];
		}
	}

	auto cross = aMat * bMat;
	hiprtFrameMatrix result;
	result.time = a.time;

	// This is bad, but i couldn't make std::copy work
	for ( int x = 0; x < 3; x++ ) {
		for ( int y = 0; y < 4; y++ ) {
			result.matrix[x][y] = cross( x, y );
		}
	}

	return result;
}

hiprtFrameMatrix getSRTMatrix( float3 translation, float4 rotation, float3 scale) {
	hiprtFrameMatrix res;

	auto Rraw = Eigen::Quaternionf(rotation.w, rotation.x, rotation.y, rotation.z).matrix();
	Eigen::Matrix4f R;
	R.setIdentity();
	for ( int i = 0; i < 3; i++ ) {
		for ( int j = 0; j < 3; j++ ) {
			R( i, j ) = Rraw( i, j );
		}
	}

	Eigen::Matrix4f S;
	S.setIdentity();
	S( 0, 0 ) = scale.x;
	S( 1, 1 ) = scale.y;
	S( 2, 2 ) = scale.z;

	Eigen::Matrix4f T;
	T.setIdentity();
	T( 0, 3 ) = translation.x;
	T( 1, 3 ) = translation.y;
	T( 2, 3 ) = translation.z;

	auto M = T * R * S;

	for ( int i = 0; i < 3; i++ ) {
		for ( int j = 0; j < 4; j++ ) {
			res.matrix[i][j] = M( i, j );
		}
	}

	res.time = 0;
	
	return res;
}

inline void loadNode(
	tinygltf::Node	 node,
	tinygltf::Model	 model,
	std::vector<Geometry>& geomData,
	hiprtContext& ctxt,
	std::vector<hiprtGeometry>& geometries,
	std::vector<hiprtFrameMatrix>& frames,
	std::vector<hiprtTransformHeader>& srtHeaders,
	hiprtFrameMatrix parentTransform = getSRTMatrix({0, 0, 0}, {0, 0, 0, 0}, {1, 1, 1}) ) {

	// Process transforms

	// Compiler! Why don't u let me use ternary!? Now I have to suffer!
	float3 translate;
	if ( node.translation.size() == 0 )
		translate = { 0, 0, 0 }; 
	else 
		translate = { 
		static_cast<float>(node.translation[0]), 
		static_cast<float>(node.translation[1]), 
		static_cast<float>(node.translation[2]) };

	float4 rotation;
	if ( node.rotation.size() == 0 )
		rotation = { 0, 0, 0, 0 };
	else
		rotation = {
			static_cast<float>( node.rotation[0] ),
			static_cast<float>( node.rotation[1] ),
			static_cast<float>( node.rotation[2] ),
			static_cast<float>( node.rotation[3] ) };

	float3 scale;
	if ( node.scale.size() == 0 )
		scale = { 1, 1, 1 };
	else
		scale = {
			static_cast<float>( node.scale[0] ), 
			static_cast<float>( node.scale[1] ), 
			static_cast<float>( node.scale[2] ) };

	hiprtFrameMatrix localTransformations = parentTransform * getSRTMatrix( translate, rotation, scale );
	localTransformations.time			  = 0;
	frames.push_back( localTransformations );

	// Process mesh
	int meshIndex = node.mesh;
	if ( meshIndex != -1 ) {
		auto mesh = model.meshes[meshIndex];
		
		for ( auto meshPrimitive : mesh.primitives ) {
			hiprtTriangleMeshPrimitive hipMesh;

			// Load indices
			hipMesh.triangleCount = model.accessors[meshPrimitive.indices].count / 3;
			hipMesh.triangleStride = sizeof( hiprtInt3 );

			auto indicesBufferView = model.bufferViews[model.accessors[meshPrimitive.indices].bufferView];

			short* rawIndices = (short*)malloc(
				hipMesh.triangleCount *
				tinygltf::GetComponentSizeInBytes( model.accessors[meshPrimitive.indices].componentType ) * 3 );
			memcpy(
				rawIndices,
				&model.buffers[indicesBufferView.buffer].data[0] + indicesBufferView.byteOffset,
				hipMesh.triangleCount *
					tinygltf::GetComponentSizeInBytes( model.accessors[meshPrimitive.indices].componentType ) * 3 );

			int* indices = (int*)malloc( hipMesh.triangleCount * sizeof( hiprtInt3 ) );
			for ( int i = 0; i < hipMesh.triangleCount * 3; i++ ) {
				indices[i] = rawIndices[i];
			}

			geomData.push_back( { nullptr, nullptr, nullptr } );
			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &hipMesh.triangleIndices ), hipMesh.triangleCount * hipMesh.triangleStride ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( hipMesh.triangleIndices ),
				indices,
				hipMesh.triangleCount * hipMesh.triangleStride ) );

			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &( geomData.back().indices ) ),
				hipMesh.triangleCount * hipMesh.triangleStride ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( geomData.back().indices ),
				indices,
				hipMesh.triangleCount * hipMesh.triangleStride ) );

			free( rawIndices );
			free( indices );

			// Load vertices

			auto accessorIter = meshPrimitive.attributes.find( "POSITION" );
			if ( accessorIter == meshPrimitive.attributes.end() )
				throw std::string( "Err: no POSITION attribute specified in primitive" );

			int accessorIndex = ( *accessorIter ).second;

			hipMesh.vertexCount	 = model.accessors[accessorIndex].count;
			hipMesh.vertexStride = tinygltf::GetComponentSizeInBytes( model.accessors[accessorIndex].componentType ) * 3;

			auto vertexBufferView = model.bufferViews[model.accessors[accessorIndex].bufferView];

			hiprtFloat3* a = (hiprtFloat3*)malloc( hipMesh.vertexCount * hipMesh.vertexStride );
			memcpy(
				a,
				&model.buffers[vertexBufferView.buffer].data[0] + vertexBufferView.byteOffset,
				hipMesh.vertexCount * hipMesh.vertexStride );

			CHECK_ORO(
				oroMalloc( reinterpret_cast<oroDeviceptr*>( &hipMesh.vertices ), hipMesh.vertexCount * hipMesh.vertexStride ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( hipMesh.vertices ), a, hipMesh.vertexCount * hipMesh.vertexStride ) );

			free( a );

			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &( geomData.back().vertices ) ),
				hipMesh.vertexCount * hipMesh.vertexStride ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( geomData.back().vertices ),
				&model.buffers[vertexBufferView.buffer].data[0] + vertexBufferView.byteOffset,
				hipMesh.vertexCount * hipMesh.vertexStride ) );

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.triangleMesh.primitive = hipMesh;

			size_t			  geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			hiprtGeometry geom;
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

			geometries.push_back( geom );
			hiprtTransformHeader header;
			// TODO: if we will implement animations, this should be changed
			header.frameCount = 1; 
			header.frameIndex = frames.size() - 1;
			srtHeaders.push_back( header );

			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );

			// Load vertex normals
			{
				auto accessorIter = meshPrimitive.attributes.find( "NORMAL" );
				if ( accessorIter == meshPrimitive.attributes.end() )
					throw std::string( "Err: no NORMAL attribute specified in primitive" );

				int accessorIndex = ( *accessorIter ).second;

				int normalCount	 = model.accessors[accessorIndex].count;
				int normalStride = tinygltf::GetComponentSizeInBytes( model.accessors[accessorIndex].componentType ) * 3;

				auto normalBufferView = model.bufferViews[model.accessors[accessorIndex].bufferView];

				a = (hiprtFloat3*)malloc( normalCount * normalStride );
				memcpy(
					a,
					&model.buffers[normalBufferView.buffer].data[0] + normalBufferView.byteOffset,
					normalCount * normalStride );

				CHECK_ORO(
					oroMalloc( reinterpret_cast<oroDeviceptr*>( &( geomData.back().normals ) ), normalCount * normalStride ) );
				CHECK_ORO(
					oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( geomData.back().normals ), a, normalCount * normalStride ) );

				free( a );
			}

		}
	}

	// TODO: Process camera

	for ( auto nodeChild : node.children ) {
		loadNode( model.nodes[nodeChild], model, geomData, ctxt, geometries, frames, srtHeaders, localTransformations );
	}
}

void IRenderEngine::loadModel(
	std::string&					   path,
	hiprtContext&					   ctxt,
	std::vector<hiprtFrameMatrix>&	   frames,
	std::vector<hiprtTransformHeader>& srtHeaders ) {
	tinygltf::Model	   model;
	tinygltf::TinyGLTF loader;
	std::string		   err;
	std::string		   warn;

	bool ret = loader.LoadASCIIFromFile( &model, &err, &warn, path );

	if ( !warn.empty() ) printf( "Warn: %s\n", warn.c_str() );

	if ( !err.empty() ) printf( "Err: %s\n", err.c_str() );

	if ( !ret ) throw std::runtime_error( "Failed to parse glTF" );

	std::vector<Geometry> geomData;

	for ( auto nodeIndex : model.scenes[model.defaultScene].nodes ) {
		auto rootNode = model.nodes[nodeIndex];

		loadNode(rootNode, model, geomData, ctxt, geometries, frames, srtHeaders);
	}

	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &gpuGeometry ), sizeof( Geometry ) * geomData.size() ) );
	CHECK_ORO(
		oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( gpuGeometry ), &(geomData[0]), sizeof( Geometry ) * geomData.size() ) );

	textureAmount			 = 1;
	Texture texturesOrigin[] = { createTexture( ctxt, { 255, 0, 0, 0 } ) };
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &textures ), textureAmount * sizeof( Texture ) ) );
	CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( textures ), texturesOrigin, textureAmount * sizeof( Texture ) ) );

	// �������� ����������
	constexpr int geomAmount			 = 2;
	constexpr int matsAmount			 = 1;
	Material	  matsOrigin[matsAmount] = { Material( 0, 0, 0, 0, 0 ) };
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &materials ), matsAmount * sizeof( Material ) ) );
	CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( materials ), matsOrigin, matsAmount * sizeof( Material ) ) );

	// ������� ����������
	int materialInd[geomAmount] = { 0, 0 };
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &materialIndices ), geomAmount * sizeof( int ) ) );
	CHECK_ORO( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( materialIndices ), materialInd, geomAmount * sizeof( int ) ) );
}

void IRenderEngine::init( int deviceIndex, int width, int height ) {
	m_res = make_hiprtInt2( width, height );

	CHECK_ORO( (oroError)oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 ) );

	CHECK_ORO( oroInit( 0 ) );
	CHECK_ORO( oroDeviceGet( &m_oroDevice, deviceIndex ) );
	CHECK_ORO( oroCtxCreate( &m_oroCtx, 0, m_oroDevice ) );

	oroDeviceProp props;
	CHECK_ORO( oroGetDeviceProperties( &props, m_oroDevice ) );

	std::cout << "hiprt ver." << HIPRT_VERSION_STR << std::endl;
	std::cout << "Executing on '" << props.name << "'" << std::endl;
	if ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
		m_ctxtInput.deviceType = hiprtDeviceNVIDIA;
	else
		m_ctxtInput.deviceType = hiprtDeviceAMD;

	m_ctxtInput.ctxt   = oroGetRawCtx( m_oroCtx );
	m_ctxtInput.device = oroGetRawDevice( m_oroDevice );
	hiprtSetLogLevel( hiprtLogLevelError );

	CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

	std::vector<hiprtFrameMatrix> frames;
	std::vector<hiprtTransformHeader> srtHeaders;
	loadModel( std::string( "testmodels/default.gltf" ), ctxt, frames, srtHeaders );

	sceneInput.instanceCount			= geometries.size();
	sceneInput.instanceMasks			= nullptr;
	sceneInput.instanceTransformHeaders = nullptr;
	CHECK_ORO( oroMalloc(
		reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceGeometries ),
		sizeof( hiprtDevicePtr ) * sceneInput.instanceCount ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( sceneInput.instanceGeometries ),
		&geometries[0],
		sizeof( hiprtDevicePtr ) * sceneInput.instanceCount ) );

	sceneInput.frameType   = hiprtFrameTypeMatrix;
	//hiprtFrameMatrix frame = getSRTMatrix( { 0, 0, 0 }, { 0, 0, 1, 0 }, { 1, 1, 1 } );
	//frame.time				 = 0;
	int frameCount = frames.size();

	CHECK_ORO(
		oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceFrames ), sizeof( hiprtFrameMatrix ) * frameCount ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ), &frames[0], sizeof( hiprtFrameMatrix ) * frameCount ) );

	sceneInput.frameCount = frameCount;

	CHECK_ORO( oroMalloc(
		reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceTransformHeaders ),
		srtHeaders.size() * sizeof( hiprtTransformHeader ) ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( sceneInput.instanceTransformHeaders ),
		&srtHeaders[0],
		srtHeaders.size() * sizeof( hiprtTransformHeader ) ) );

	size_t		   sceneTempSize;
	hiprtDevicePtr sceneTemp;
	CHECK_HIPRT( hiprtGetSceneBuildTemporaryBufferSize( ctxt, sceneInput, options, sceneTempSize ) );
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneTemp ), sceneTempSize ) );

	CHECK_HIPRT( hiprtCreateScene( ctxt, sceneInput, options, scene ) );
	CHECK_HIPRT( hiprtBuildScene( ctxt, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, scene ) );

	buildTraceKernelFromBitcode( ctxt, "../common/Kernels.h", "SceneIntersectionKernel", func );

	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );

	CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( sceneTemp ) ) );
}

void IRenderEngine::onResize( int width, int height ) {
	m_res = make_hiprtInt2( width, height );

	CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( pixels ) ) );
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &pixels ), m_res.x * m_res.y * 4 ) );
}

bool IRenderEngine::readSourceCode(
	const std::filesystem::path& path, std::string& sourceCode, std::optional<std::vector<std::filesystem::path>> includes ) {
	std::fstream f( path );
	if ( f.is_open() ) {
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = static_cast<size_t>( f.tellg() );
		f.seekg( 0, std::fstream::beg );
		if ( includes ) {
			sourceCode.clear();
			std::string line;
			while ( std::getline( f, line ) ) {
				if ( line.find( "#include" ) != std::string::npos ) {
					size_t		pa	= line.find( "<" );
					size_t		pb	= line.find( ">" );
					std::string buf = line.substr( pa + 1, pb - pa - 1 );
					includes.value().push_back( buf );
					sourceCode += line + '\n';
				}
				sourceCode += line + '\n';
			}
		} else {
			sourceCode.resize( size, ' ' );
			f.read( &sourceCode[0], size );
		}
		f.close();
	} else
		return false;
	return true;
}

void IRenderEngine::buildTraceKernelFromBitcode(
	hiprtContext				   ctxt,
	const char*					   path,
	const char*					   functionName,
	oroFunction&				   functionOut,
	std::vector<const char*>*	   opts,
	std::vector<hiprtFuncNameSet>* funcNameSets,
	int							   numGeomTypes,
	int							   numRayTypes ) {
	std::vector<const char*>		   options;
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;

	if ( !readSourceCode( path, sourceCode, includeNamesData ) ) {
		std::cerr << "Unable to find file '" << path << "'" << std::endl;
		;
		exit( EXIT_FAILURE );
	}

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( int i = 0; i < includeNamesData.size(); i++ ) {
		if ( !readSourceCode( std::string( "../../" ) / includeNamesData[i], headersData[i] ) ) {
			if ( !readSourceCode( std::string( "../" ) / includeNamesData[i], headersData[i] ) ) {
				std::cerr << "Failed to find header file '" << includeNamesData[i] << "' in path ../ or ../../!" << std::endl;
				exit( EXIT_FAILURE );
			}
		}
		includeNames.push_back( includeNamesData[i].string().c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	if ( opts ) {
		for ( const auto o : *opts )
			options.push_back( o );
	}

	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
	if ( isAmd ) {
		options.push_back( "-fgpu-rdc" );
		options.push_back( "-Xclang" );
		options.push_back( "-disable-llvm-passes" );
		options.push_back( "-Xclang" );
		options.push_back( "-mno-constructor-aliases" );
	} else {
		options.push_back( "--device-c" );
		options.push_back( "-arch=compute_60" );
	}
	options.push_back( "-std=c++17" );
	options.push_back( "-I../" );
	options.push_back( "-I../../" );

	orortcProgram prog;
	CHECK_ORORTC( orortcCreateProgram(
		&prog, sourceCode.data(), path, static_cast<int>( headers.size() ), headers.data(), includeNames.data() ) );
	CHECK_ORORTC( orortcAddNameExpression( prog, functionName ) );

	orortcResult e = orortcCompileProgram( prog, static_cast<int>( options.size() ), options.data() );
	if ( e != ORORTC_SUCCESS ) {
		size_t logSize;
		CHECK_ORORTC( orortcGetProgramLogSize( prog, &logSize ) );

		if ( logSize ) {
			std::string log( logSize, '\0' );
			orortcGetProgramLog( prog, &log[0] );
			std::cerr << log << std::endl;
		}
		exit( EXIT_FAILURE );
	}

	std::string bitCodeBinary;
	size_t		size = 0;
	if ( isAmd )
		CHECK_ORORTC( orortcGetBitcodeSize( prog, &size ) );
	else
		CHECK_ORORTC( orortcGetCodeSize( prog, &size ) );
	assert( size != 0 );

	bitCodeBinary.resize( size );
	if ( isAmd )
		CHECK_ORORTC( orortcGetBitcode( prog, (char*)bitCodeBinary.data() ) );
	else
		CHECK_ORORTC( orortcGetCode( prog, (char*)bitCodeBinary.data() ) );

	hiprtApiFunction function;
	CHECK_HIPRT( hiprtBuildTraceKernelsFromBitcode(
		ctxt,
		1,
		&functionName,
		path,
		bitCodeBinary.data(),
		size,
		numGeomTypes,
		numRayTypes,
		funcNameSets != nullptr ? funcNameSets->data() : nullptr,
		&function,
		false ) );

	functionOut = *reinterpret_cast<oroFunction*>( &function );
}

void IRenderEngine::launchKernel( oroFunction func, int nx, int ny, void** args ) { launchKernel( func, nx, ny, 8, 8, args ); }

void IRenderEngine::launchKernel( oroFunction func, int nx, int ny, int bx, int by, void** args ) {
	hiprtInt3 nb;
	nb.x = ( nx + bx - 1 ) / bx;
	nb.y = ( ny + by - 1 ) / by;
	CHECK_ORO( oroModuleLaunchKernel( func, nb.x, nb.y, 1, bx, by, 1, 0, 0, args, 0 ) );
}