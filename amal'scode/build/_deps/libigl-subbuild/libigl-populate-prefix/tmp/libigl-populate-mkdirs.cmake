# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-src")
  file(MAKE_DIRECTORY "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-src")
endif()
file(MAKE_DIRECTORY
  "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-build"
  "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix"
  "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix/tmp"
  "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp"
  "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix/src"
  "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/gowtham/Desktop/Projects/amal-arap-poyline copy/amal'scode/build/_deps/libigl-subbuild/libigl-populate-prefix/src/libigl-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
