@echo off
setlocal
set GXX=%~dp0..\..\data\mingw64\bin\g++.exe
"%GXX%" -std=c++26 -freflection -O2 -Wall -Wextra "%~dp0json_struct_reflection.cpp" -o "%~dp0json_struct_reflection.exe" || exit /b 1
"%~dp0json_struct_reflection.exe" %*
