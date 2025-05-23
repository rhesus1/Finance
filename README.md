Compile : g++ src/*.cpp main.cpp -fopenmp -std=c++17 -I include -I /c/Boost/include/boost-1_88 -I /mingw64/include -L /mingw64/lib -lcurl -O3 -o main

Note: Ensure installation of Boost, libcurl and nlohmann-json
