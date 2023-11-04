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
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../contrib/tiny_gltf.h"

#include "common/Base.h"
#include "common/Common.h"
#include <gl/freeglut.h>

#include <future>
#include <iostream>
#include <istream>
#include <thread>

class RenderEngine : public IRenderEngine {
  public:
	void run( u8* data, int time ) {
		renderingMutex.lock();

		float3* debug = (float3*)malloc( sizeof( float3 ) * 4 );
		float3* gpuDebug;
		CHECK_ORO( oroMalloc( reinterpret_cast<oroDeviceptr*>( &gpuDebug ), sizeof( float3 ) * 4 ) );

		Camera cam( make_float3( 7.35889, -6.92579, 4.95831 ), 
			make_float3( 63.5593, 0, 46.6919 ), 2 );
		//float3 vec	  = {0, 1, 1};
		//vec			 = cam.getRotatedVector( vec );
		void*  args[] = { &scene, &pixels, &m_res, &gpuGeometry, &textures, &materials, &materialIndices, &cam, &time, &gpuDebug };
		launchKernel( func, m_res.x, m_res.y, args );

		CHECK_ORO( oroMemcpyDtoH( data, reinterpret_cast<oroDeviceptr>( pixels ), m_res.x * m_res.y * 4 ) );
		renderingMutex.unlock();

		CHECK_ORO( oroMemcpyDtoH( debug, reinterpret_cast<oroDeviceptr>( gpuDebug ), sizeof( float3 ) * 4 ) );

		for ( int i = 0; i < 4; i++ ) {
			float3 bam = debug[i];

			bam = bam;
		}
	}
};

int			 timeee  = 0;
int			 width = 1000, height = 1000;
u8*			 data;
RenderEngine renderEngine;

void display();
void resize( int w, int h );
void init();

void resize( int w, int h ) {
	w = width, h = height;
	//renderEngine.renderingMutex.lock();
	width = w, height = h;
	//renderEngine.onResize( w, h );
	glutReshapeWindow( width, height );
	//data = (u8*)realloc( data, width * height * 4 );
	//renderEngine.renderingMutex.unlock();
}

void display() {
	glClearColor( 0, 0, 0, 0.0 );
	glClear( GL_COLOR_BUFFER_BIT ); // clear display window

	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	gluOrtho2D( 0.0, width, 0.0, height);

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glColor3f( 1.0, 1.0, 1.0 );

	glPointSize( 1.0f );
	glBegin( GL_POINTS );
	for ( int w = 0; w < width; w++ ) {
		for ( int h = 0; h < height; h++ ) {
			glColor4ub(
				data[( w + width * h ) * 4],
				data[( w + width * h ) * 4 + 1],
				data[( w + width * h ) * 4 + 2],
				data[( w + width * h ) * 4 + 3] );
			glVertex2i( w, height - h );

		}
	}
	glEnd();
	glFlush();
}

// Program to create an empty Widdow
void init() {
	glutInitDisplayMode( GLUT_RGB ); // Line C
	glutInitWindowSize( width, height );
	glutCreateWindow( "CPPRay" );
}

int main( int argc, char** argv ) {
	data = (u8*)malloc( width * height * 4 );
	renderEngine.init( 0, width, height );

	glutInit( &argc, argv ); // Line A
	init();					 // Line B
	glutReshapeFunc( resize );
	glutDisplayFunc( display );

	auto future1 = std::async( std::launch::async, [] {
		//while ( true )
		{
			renderEngine.run( data, timeee % 360 );
			glutPostRedisplay();

			timeee++;
			Sleep( 50 );
		}
	} );

	glutMainLoop();

	return 0;
}