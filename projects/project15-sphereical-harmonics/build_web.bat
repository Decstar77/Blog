@echo off
setlocal enabledelayedexpansion

rem Builds the spherical harmonics demo to WebAssembly and copies the bundle
rem into site/demos/spherical-harmonics/ where nginx serves it.
rem
rem   build_web.bat          configure, build, deploy
rem   build_web.bat clean    delete build-web/ first (forces a raylib refetch)

set "PROJDIR=%~dp0"
set "BUILDDIR=%PROJDIR%build-web"
set "DEPLOYDIR=%PROJDIR%..\..\site\demos\spherical-harmonics"

if /i "%~1"=="clean" (
    echo [clean] removing %BUILDDIR%
    if exist "%BUILDDIR%" rmdir /s /q "%BUILDDIR%"
)

rem ---- Emscripten -------------------------------------------------------
if not defined EMSDK set "EMSDK=C:\Projects\2026\Emscripten\emsdk"

if not exist "%EMSDK%\emsdk_env.bat" (
    echo ERROR: no emsdk at "%EMSDK%".
    echo Set the EMSDK environment variable to your emsdk checkout and retry.
    exit /b 1
)

rem Deliberately not using emsdk_env.bat. It delegates to `emsdk construct_env`,
rem which sniffs the parent shell, and on this machine it decides it is talking
rem to bash and emits `export PATH=/c/...` instead of writing the
rem emsdk_set_env.bat that the cmd path relies on. Nothing then gets set and the
rem failure only shows up on a cold configure. These four variables are all the
rem toolchain actually needs.
set "EMROOT=%EMSDK%\upstream\emscripten"
set "PATH=%EMROOT%;%EMSDK%;%PATH%"

rem .emscripten resolves node, llvm and binaryen relative to EM_CONFIG itself.
set "EM_CONFIG=%EMSDK%\.emscripten"

rem Version-stamped directories, so glob rather than hardcode.
for /d %%D in ("%EMSDK%\python\*") do if exist "%%D\python.exe" set "EMSDK_PYTHON=%%D\python.exe"
for /d %%D in ("%EMSDK%\node\*") do if exist "%%D\bin\node.exe" set "EMSDK_NODE=%%D\bin\node.exe"

if not exist "%EMROOT%\emcmake.bat" (
    echo ERROR: no emcmake.bat under "%EMROOT%".
    exit /b 1
)

rem ---- Generator -------------------------------------------------------
rem emcmake cannot drive the Visual Studio generator, so pick something that
rem takes a clang-style compiler.
where ninja >nul 2>&1
if errorlevel 1 (
    if exist "C:\Projects\2026\Heisenburg\mingw64\bin\ninja.exe" (
        set "PATH=C:\Projects\2026\Heisenburg\mingw64\bin;!PATH!"
        set "GENERATOR=Ninja"
    ) else (
        set "GENERATOR=MinGW Makefiles"
    )
) else (
    set "GENERATOR=Ninja"
)
echo [build] generator: !GENERATOR!

rem ---- Configure -------------------------------------------------------
rem FetchContent pulls raylib 5.5 and builds it with PLATFORM=Web, so the
rem first configure is slow. Skipped once the cache exists.
if not exist "%BUILDDIR%\CMakeCache.txt" (
    echo [build] configuring ^(fetches and builds raylib, this takes a while^)
    call "%EMROOT%\emcmake.bat" cmake -S "%PROJDIR%." -B "%BUILDDIR%" -G "!GENERATOR!" -DCMAKE_BUILD_TYPE=Release
    if errorlevel 1 exit /b 1
)

rem CMake passes --shell-file as a link option, which the generator does not
rem track as an input. Without this, edits to web/shell.html are silently
rem ignored because the link step still looks up to date.
if exist "%BUILDDIR%\SphericalHarmonics.html" del /q "%BUILDDIR%\SphericalHarmonics.html"

echo [build] compiling
call cmake --build "%BUILDDIR%"
if errorlevel 1 exit /b 1

rem ---- Deploy ----------------------------------------------------------
rem The generated page is renamed so the demo is reachable as a directory URL.
if not exist "%DEPLOYDIR%" mkdir "%DEPLOYDIR%"
copy /y "%BUILDDIR%\SphericalHarmonics.html" "%DEPLOYDIR%\index.html" >nul || exit /b 1
for %%F in (js wasm data) do (
    copy /y "%BUILDDIR%\SphericalHarmonics.%%F" "%DEPLOYDIR%\" >nul || exit /b 1
)

echo [done] deployed to site\demos\spherical-harmonics\
echo        preview: python -m http.server 8899 --directory "%PROJDIR%..\..\site"
echo                 then open http://localhost:8899/project-spherical-harmonics.html

endlocal
