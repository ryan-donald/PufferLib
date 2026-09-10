#!/usr/bin/env bash
# Run from the PufferLib root; requires Raylib 5.5 (build.sh downloads it).
set -euo pipefail
mkdir -p build
includes=(-Isrc -Iraylib-5.5_linux_amd64/include)
libs=(raylib-5.5_linux_amd64/lib/libraylib.a -lm -ldl -lpthread)
"${CC:-gcc}" -std=c11 -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer "${includes[@]}" tests/test_bomber.c "${libs[@]}" -o build/test_bomber
./build/test_bomber
"${CXX:-g++}" -x c++ -std=c++17 -O2 "${includes[@]}" tests/test_bomber.c \
    -x none "${libs[@]}" -o build/test_bomber_cpp
./build/test_bomber_cpp
"${CC:-gcc}" -std=c11 -O3 "${includes[@]}" tests/bench_bomber.c "${libs[@]}" -o build/bench_bomber
./build/bench_bomber 1000
