find_path(NOVA_ROOT_DIR
  NAMES include/nova/nova.hpp
)

find_path(NOVA_INCLUDE_DIR
  NAMES nova/nova.hpp
  HINTS ${NOVA_ROOT_DIR}/include
)

find_file(NOVA_DEPS
  NAMES nova_dependencies
  HINTS ${NOVA_ROOT_DIR}
)

find_library(NOVA_LIBRARY
  NAMES nova libnova
  HINTS ${NOVA_ROOT_DIR}/lib
)

file(STRINGS "${NOVA_DEPS}" DEPS_HEADERS_AND_LIBS)
list(GET DEPS_HEADERS_AND_LIBS 0 NOVA_DEPS_HEADERS)
list(GET DEPS_HEADERS_AND_LIBS 1 NOVA_DEPS_LIBS)
list(GET DEPS_HEADERS_AND_LIBS 2 NOVA_USE_PHMAP)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NOVA
  FOUND_VAR NOVA_FOUND
  REQUIRED_VARS NOVA_LIBRARY NOVA_INCLUDE_DIR NOVA_DEPS_HEADERS NOVA_DEPS_LIBS
)

mark_as_advanced(NOVA_ROOT_DIR
  NOVA_LIBRARY
  NOVA_INCLUDE_DIR
  NOVA_DEPS_HEADERS
  NOVA_DEPS_LIBS
)

if (NOVA_FOUND)
  message(STATUS "Found valid NOVA version:")
  message(STATUS "  NOVA root dir: ${NOVA_ROOT_DIR}")
  message(STATUS "  NOVA include dir: ${NOVA_INCLUDE_DIR}")
  message(STATUS "  NOVA libraries: ${NOVA_LIBRARY}")
  message(STATUS "  NOVA dependency include dirs: ${NOVA_DEPS_HEADERS}")
  message(STATUS "  NOVA dependency library dirs: ${NOVA_DEPS_LIBS}")
endif ()
