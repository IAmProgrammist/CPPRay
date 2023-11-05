echo Create folders

mkdir "%~dp0src\dist\bin\Debug"
mkdir "%~dp0src\dist\bin\Release"
mkdir "%~dp0tmp"
mkdir "%~dp0tmp\hiprt"
mkdir "%~dp0hiprt"

echo Download HIPRT and unpack

bitsadmin.exe /transfer "DownloadHIPRT" https://gpuopen.com/download/hiprt/hiprtSdk-2.1.6fc8ff0.zip "%~dp0/tmp/hiprt.zip"
tar -x -f "%~dp0tmp\hiprt.zip" -C "%~dp0tmp\hiprt"

echo Copy HIPRT to project 

xcopy /s /e "%~dp0tmp\hiprt\hiprt" "%~dp0hiprt"

echo Copy dlls for Debug build

xcopy "%~dp0contrib\SFML-2.6.1\bin\openal32.dll" "%~dp0src\dist\bin\Debug\openal32.dll"*
xcopy "%~dp0contrib\SFML-2.6.1\bin\sfml-graphics-d-2.dll" "%~dp0src\dist\bin\Debug\sfml-graphics-d-2.dll"*
xcopy "%~dp0contrib\SFML-2.6.1\bin\sfml-window-d-2.dll" "%~dp0src\dist\bin\Debug\sfml-window-d-2.dll"*
xcopy "%~dp0contrib\SFML-2.6.1\bin\sfml-system-d-2.dll" "%~dp0src\dist\bin\Debug\sfml-system-d-2.dll"*
xcopy /s /e "%~dp0\hiprt\win" "%~dp0src\dist\bin\Debug"

echo Copy dlls for Release build


xcopy "%~dp0contrib\SFML-2.6.1\bin\openal32.dll" "%~dp0src\dist\bin\Release\openal32.dll"*
xcopy "%~dp0contrib\SFML-2.6.1\bin\sfml-graphics-2.dll" "%~dp0src\dist\bin\Release\sfml-graphics-2.dll"*
xcopy "%~dp0contrib\SFML-2.6.1\bin\sfml-window-2.dll" "%~dp0src\dist\bin\Release\sfml-window-2.dll"*
xcopy "%~dp0contrib\SFML-2.6.1\bin\sfml-system-2.dll" "%~dp0src\dist\bin\Release\sfml-system-2.dll"*
xcopy /s /e "%~dp0\hiprt\win" "%~dp0src\dist\bin\Release"

echo Delete temporary files

del /s /q "%~dp0tmp"
rmdir /s /q "%~dp0tmp"