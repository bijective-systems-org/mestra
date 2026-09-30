cmake_minimum_required(VERSION 3.16)
project(mestra_cpp VERSION 0.1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MESTRA_BUILD_TESTS "Build the mestra test suite" ON)
# A file is untrusted input, so the suite is meant to be run under the
# sanitizers from time to time as well as plain.
option(MESTRA_SANITIZE "Build with the address and undefined sanitizers" OFF)

# Everything built here -- the library, the tool, the tests and the
# examples -- compiles under these and is warning-free.  They are
# spelled per compiler family.  MESTRA_FP_FLAGS is the one that
# matters for conformance: section 27 fixes the summation order of the
# affine callable and the corpus compares float64 results bit for bit,
# so no multiply and add may be fused into a single rounding step.
# GCC and clang are told so with -ffp-contract=off; MSVC does not
# contract under /fp:precise (its default since Visual Studio 2022),
# and src/affine.cpp says it again with a pragma for each of them.
if(MSVC)
  set(MESTRA_WARNINGS /W4 /permissive-)
  set(MESTRA_FP_FLAGS /fp:precise)
else()
  set(MESTRA_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion
                      -Wsign-conversion)
  set(MESTRA_FP_FLAGS -ffp-contract=off)
endif()
set(MESTRA_SANITIZER_FLAGS "")
if(MESTRA_SANITIZE)
  set(MESTRA_SANITIZER_FLAGS
      -fsanitize=address,undefined -fno-omit-frame-pointer
      -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(HDF5 REQUIRED COMPONENTS C HL)

add_library(mestra
  src/sha256.cpp
  src/units.cpp
  src/names.cpp
  src/value.cpp
  src/dataset.cpp
  src/h5.cpp
  src/layout.cpp
  src/codec.cpp
  src/read.cpp
  src/write.cpp
  src/append.cpp
  src/validate.cpp
  src/weights.cpp
  src/post.cpp
  src/callable.cpp
  src/affine.cpp
  src/evaluate.cpp
)

target_include_directories(mestra
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(TARGET HDF5::HDF5)
  target_link_libraries(mestra PUBLIC HDF5::HDF5)
else()
  target_include_directories(mestra PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(mestra PUBLIC ${HDF5_LIBRARIES})
endif()
if(HDF5_HL_LIBRARIES)
  target_link_libraries(mestra PUBLIC ${HDF5_HL_LIBRARIES})
elseif(TARGET HDF5::HDF5_HL)
  target_link_libraries(mestra PUBLIC HDF5::HDF5_HL)
endif()

target_compile_options(mestra PRIVATE
  ${MESTRA_WARNINGS} ${MESTRA_FP_FLAGS} ${MESTRA_SANITIZER_FLAGS})

add_executable(mestra-cli tools/mestra_cli.cpp)
target_link_libraries(mestra-cli PRIVATE mestra)
target_compile_options(mestra-cli PRIVATE
  ${MESTRA_WARNINGS} ${MESTRA_FP_FLAGS} ${MESTRA_SANITIZER_FLAGS})

add_subdirectory(examples)

if(MESTRA_BUILD_TESTS)
  enable_testing()

  # `unit` is the pure C++ checks; its own file says what they are.
  add_executable(mestra-unit tests/unit_tests.cpp)
  target_link_libraries(mestra-unit PRIVATE mestra)
  target_include_directories(mestra-unit PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_options(mestra-unit PRIVATE
    ${MESTRA_WARNINGS} ${MESTRA_FP_FLAGS} ${MESTRA_SANITIZER_FLAGS})
  add_test(NAME unit COMMAND mestra-unit)

  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    # Every case of vectors/cases through mestra-cli: the validator
    # outcome and the shape of its output, every support id, every
    # probe, every codec round trip, every worked evaluation, a lazy
    # row read, and read-write-compare for the cases that validate.
    # Three corpus files are generated rather than committed, because
    # of their size; run `python vectors/generate.py --on-demand`
    # once before ctest, and not again while a test is reading
    # vectors/.
    add_test(NAME corpus COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_corpus.py
      --cli $<TARGET_FILE:mestra-cli>
      --vectors ${CMAKE_CURRENT_SOURCE_DIR}/../vectors)
    # Conventions section 7: the metadata open decides the nine
    # structural rules from what it is allowed to read, and names the
    # same ones the whole read names.  Checked on every file there is.
    add_test(NAME metadata_open COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_metadata.py
      --cli $<TARGET_FILE:mestra-cli>
      --vectors ${CMAKE_CURRENT_SOURCE_DIR}/../vectors)
    set_tests_properties(metadata_open PROPERTIES TIMEOUT 900)

    # Sections 12 and 29 forbid interpreting /private and not copying
    # it, and no corpus case carries one, so the round trip that keeps
    # it is checked here.
    add_test(NAME private_roundtrip COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_private.py
      --cli $<TARGET_FILE:mestra-cli>
      --vectors ${CMAKE_CURRENT_SOURCE_DIR}/../vectors)

    # Files that are legal HDF5 and are not this format.  They say
    # nothing about what a conforming file means; they say that a
    # reader answers for a file it did not write.
    # The corpus's own hostile subset, whose contract section 30
    # gives: the required identifiers must appear, more are allowed,
    # and validate, info and read must all refuse inside the timeout.
    # Two of its files are generated rather than committed; see
    # tests/run_shared_hostile.py.
    add_test(NAME shared_hostile COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_shared_hostile.py
      --cli $<TARGET_FILE:mestra-cli>
      --vectors ${CMAKE_CURRENT_SOURCE_DIR}/../vectors)
    set_tests_properties(shared_hostile PROPERTIES TIMEOUT 900)

    add_test(NAME hostile COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_hostile.py
      --cli $<TARGET_FILE:mestra-cli>
      --dir ${CMAKE_CURRENT_SOURCE_DIR}/tests/hostile)
    set_tests_properties(hostile PROPERTIES TIMEOUT 600)

    # The seven examples, each run and compared to the "Expected
    # output" block of the README beside its source, so that a change
    # to this API that a document did not follow fails here.
    add_test(NAME examples COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_examples.py
      --bin ${MESTRA_EXAMPLE_BINARIES}
      --examples ${CMAKE_CURRENT_SOURCE_DIR}/../docs/examples)
  endif()
endif()

install(TARGETS mestra mestra-cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
