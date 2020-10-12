cmake_minimum_required(VERSION 3.15)
project(untitled)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(CONSOLE)

add_executable(untitled
        src/cmain.cpp
        src/ineuron.h src/ineuron.cpp
        src/supervisor.h src/supervisor.cpp
        src/net.h src/net.cpp
        src/cmain.cpp src/environment.h)