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

void IRenderEngine::loadModel( std::string& path, hiprtContext& ctxt, int sceneIndex ) {
	/* mesh.triangleCount = 4;
	mesh.triangleStride	  = sizeof( hiprtInt3 );
	int triangleIndices[] = { 2, 1, 0, 0, 1, 3, 3, 2, 0, 1, 2, 3 };
	CHECK_ORO(
		oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.triangleIndices ), mesh.triangleCount * sizeof( hiprtInt3 ) ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( mesh.triangleIndices ), triangleIndices, mesh.triangleCount * sizeof( hiprtInt3 ) ) );

	mesh.vertexCount	   = 4;
	mesh.vertexStride	   = sizeof( hiprtFloat3 );
	hiprtFloat3 vertices[] = { { 1.45, -0.3, -0.8 }, { -1, -1, -1 }, { -0.2, 1, -0.46 }, { 0.3, -0.44, 1 } };
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &mesh.vertices ), mesh.vertexCount * sizeof( hiprtFloat3 ) ) );
	CHECK_ORO(
		oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( mesh.vertices ), vertices, mesh.vertexCount * sizeof( hiprtFloat3 ) ) );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.triangleMesh.primitive = mesh;

	size_t geomTempSize;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

	CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );*/
	tinygltf::Model		model;
	tinygltf::TinyGLTF loader;
	std::string err;
	std::string warn;

	bool ret = loader.LoadASCIIFromFile( &model, &err, &warn, path );

	if ( !warn.empty() )
		printf( "Warn: %s\n", warn.c_str() );

	if ( !err.empty() )
		printf( "Err: %s\n", err.c_str() );

	if ( !ret )
		throw std::runtime_error( "Failed to parse glTF" );

	for ( auto mesh : model.meshes ) {
		for ( auto meshPrimitive : mesh.primitives ) {
			hiprtTriangleMeshPrimitive hipMesh;

			// Load indices
			hipMesh.triangleCount  = model.accessors[meshPrimitive.indices].count;
			hipMesh.triangleStride = tinygltf::GetComponentSizeInBytes( model.accessors[meshPrimitive.indices].componentType );

			auto indicesBufferView = model.bufferViews[model.accessors[meshPrimitive.indices].bufferView];

			CHECK_ORO( oroMalloc(
				reinterpret_cast<oroDeviceptr*>( &hipMesh.triangleIndices ), hipMesh.triangleCount * hipMesh.triangleStride ) );
			CHECK_ORO( oroMemcpyHtoD(
				reinterpret_cast<oroDeviceptr>( hipMesh.triangleIndices ),
				&model.buffers[indicesBufferView.buffer].data[0] + indicesBufferView.byteOffset,
				hipMesh.triangleCount * hipMesh.triangleStride ) );


			// Load vertices
			auto accessorIter = meshPrimitive.attributes.find( "POSITION" );
			if ( accessorIter != meshPrimitive.attributes.end() ) {
				int accessorIndex = ( *accessorIter ).second;

				hipMesh.vertexCount	   = model.accessors[accessorIndex].count;
				hipMesh.vertexStride   = tinygltf::GetComponentSizeInBytes( model.accessors[accessorIndex].componentType ) * 3;

				auto vertexBufferView  = model.bufferViews[model.accessors[accessorIndex].bufferView];

				CHECK_ORO( oroMalloc(
					reinterpret_cast<oroDeviceptr*>( &hipMesh.vertices ), hipMesh.vertexCount * hipMesh.vertexStride ) );
				CHECK_ORO( oroMemcpyHtoD(
					reinterpret_cast<oroDeviceptr>( hipMesh.vertices ),
					&model.buffers[vertexBufferView.buffer].data[0] + vertexBufferView.byteOffset,
					hipMesh.vertexCount * hipMesh.vertexStride ) );
			} else {
				throw std::string( "Err: no POSITION attribute specified in primitive");
			}

			hiprtGeometryBuildInput geomInput;
			geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
			geomInput.triangleMesh.primitive = hipMesh;

			size_t geomTempSize;
			hiprtDevicePtr	  geomTemp;
			hiprtBuildOptions options;
			options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
			CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
			CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geomTemp ), geomTempSize ) );

			hiprtGeometry geom;
			CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
			CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

			geometries.push_back( geom );

			CHECK_ORO( oroFree( reinterpret_cast<oroDeviceptr>( geomTemp ) ) );
		}
	}

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


	loadModel( std::string("model.gltf"), ctxt );

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

	hiprtFrameSRT frame;
	constexpr int frameCount = 1;
	frame.translation		 = { 0, 0, 0 };
	frame.rotation			 = { 1, 0, 0, 0 };
	frame.scale				 = { 1, 1, 1 };
	frame.time				 = 0;

	CHECK_ORO(
		oroMalloc( reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceFrames ), sizeof( hiprtFrameSRT ) * frameCount ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( sceneInput.instanceFrames ), &frame, sizeof( hiprtFrameSRT ) * frameCount ) );

	sceneInput.frameCount = frameCount;

	/* hiprtTransformHeader headers[1];
	headers[0].frameIndex = 0;
	headers[0].frameCount = frameCount;
	CHECK_ORO( oroMalloc(
		reinterpret_cast<oroDeviceptr*>( &sceneInput.instanceTransformHeaders ),
		sceneInput.instanceCount * sizeof( hiprtTransformHeader ) ) );
	CHECK_ORO( oroMemcpyHtoD(
		reinterpret_cast<oroDeviceptr>( sceneInput.instanceTransformHeaders ),
		headers,
		sceneInput.instanceCount * sizeof( hiprtTransformHeader ) ) );*/

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