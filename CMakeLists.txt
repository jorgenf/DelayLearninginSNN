cmake_minimum_required(VERSION 3.4)
project(DelayLearninginSNN)
#set(EXTERNALS "${PROJECT_SOURCE_DIR}/externals")
#add_subdirectory(${EXTERNALS}/pybind11)
add_subdirectory(pybind11)

pybind11_add_module(update_states main.cpp)