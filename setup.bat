:: Create folders

mkdir "%~dp0src\dist\bin\Debug"
mkdir "%~dp0src\dist\bin\Release"
mkdir "%~dp0tmp"
mkdir "%~dp0tmp\hiprt"

:: Download HIPRT and unpack

bitsadmin.exe /transfer "DownloadHIPRT" https://gpuopen.com/download/hiprt/hiprtSdk-2.1.6fc8ff0.zip "%~dp0/tmp/hiprt.zip"
tar -x -f "%~dp0tmp\hiprt.zip" -C "%~dp0tmp\hiprt"

:: Copy HIPRT to project 

echo n|xcopy /s /e "%~dp0\tmp\hiprt\hiprt" "%~dp0\hiprt"

:: Copy dlls for Debug build

xcopy "%~dp0contrib\freeglut\bin\x64\freeglut.dll" "%~dp0src\dist\bin\Debug\freeglut.dll"*
xcopy "%~dp0contrib\glew-2.2.0\bin\Release\x64\glew32.dll" "%~dp0src\dist\bin\Debug\glew32.dll"*
xcopy /s /e "%~dp0\hiprt\win" "%~dp0src\dist\bin\Debug"

:: Copy dlls for Release build

xcopy "%~dp0contrib\freeglut\bin\x64\freeglut.dll" "%~dp0src\dist\bin\Release\freeglut.dll"*
xcopy "%~dp0contrib\glew-2.2.0\bin\Release\x64\glew32.dll" "%~dp0src\dist\bin\Release\glew32.dll"*
xcopy /s /e "%~dp0\hiprt\win" "%~dp0src\dist\bin\Release"

:: Delete temporary files

del /s /q "%~dp0tmp"
rmdir /s /q "%~dp0tmp"