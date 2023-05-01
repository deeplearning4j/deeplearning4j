# - Find Dwarf
# Find the dwarf.h header from elf utils
#
#  DWARF_INCLUDE_DIR - where to find dwarf.h, etc.
#  DWARF_LIBRARIES   - List of libraries when using elf utils.
#  DWARF_FOUND       - True if fdo found.

message(STATUS "Checking availability of DWARF and ELF development libraries")

INCLUDE(CheckLibraryExists)

if (DWARF_INCLUDE_DIR AND LIBDW_INCLUDE_DIR AND DWARF_LIBRARY AND ELF_LIBRARY)
    # Already in cache, be silent
    set(DWARF_FIND_QUIETLY TRUE)
endif (DWARF_INCLUDE_DIR AND LIBDW_INCLUDE_DIR AND DWARF_LIBRARY AND ELF_LIBRARY)

find_path(DWARF_INCLUDE_DIR dwarf.h
        /usr/include
        /usr/local/include
        /usr/include/libdwarf
        ~/usr/local/include
        )

find_path(LIBDW_INCLUDE_DIR elfutils/libdw.h
        /usr/include
        /usr/local/include
        ~/usr/local/include
        )

find_library(DWARF_LIBRARY
        NAMES dw dwarf
        PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64 ~/usr/local/lib ~/usr/local/lib64
        )

find_library(ELF_LIBRARY
        NAMES elf
        PATHS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64 ~/usr/local/lib ~/usr/local/lib64
        )

if (DWARF_INCLUDE_DIR AND LIBDW_INCLUDE_DIR AND DWARF_LIBRARY AND ELF_LIBRARY)
    set(DWARF_FOUND TRUE)
    set(DWARF_LIBRARIES ${DWARF_LIBRARY} ${ELF_LIBRARY})

    set(CMAKE_REQUIRED_LIBRARIES ${DWARF_LIBRARIES})
    # check if libdw have the dwfl_module_build_id routine, i.e. if it supports the buildid
    # mechanism to match binaries to detached debug info sections (the -debuginfo packages
    # in distributions such as fedora). We do it against libelf because, IIRC, some distros
    # include libdw linked statically into libelf.
    check_library_exists(elf dwfl_module_build_id "" HAVE_DWFL_MODULE_BUILD_ID)
else (DWARF_INCLUDE_DIR AND LIBDW_INCLUDE_DIR AND DWARF_LIBRARY AND ELF_LIBRARY)
    set(DWARF_FOUND FALSE)
    set(DWARF_LIBRARIES)
endif (DWARF_INCLUDE_DIR AND LIBDW_INCLUDE_DIR AND DWARF_LIBRARY AND ELF_LIBRARY)

if (DWARF_FOUND)
    if (NOT DWARF_FIND_QUIETLY)
        message(STATUS "Found dwarf.h header: ${DWARF_INCLUDE_DIR}")
        message(STATUS "Found elfutils/libdw.h header: ${LIBDW_INCLUDE_DIR}")
        message(STATUS "Found libdw library: ${DWARF_LIBRARY}")
        message(STATUS "Found libelf library: ${ELF_LIBRARY}")
    endif (NOT DWARF_FIND_QUIETLY)
else (DWARF_FOUND)
    if (DWARF_FIND_REQUIRED)
        # Check if we are in a Red Hat (RHEL) or Fedora system to tell
        # exactly which packages should be installed. Please send
        # patches for other distributions.
        find_path(FEDORA fedora-release /etc)
        find_path(REDHAT redhat-release /etc)
        if (FEDORA OR REDHAT)
            if (NOT DWARF_INCLUDE_DIR OR NOT LIBDW_INCLUDE_DIR)
                message(STATUS "Please install the elfutils-devel package")
            endif (NOT DWARF_INCLUDE_DIR OR NOT LIBDW_INCLUDE_DIR)
            if (NOT DWARF_LIBRARY)
                message(STATUS "Please install the elfutils-libs package")
            endif (NOT DWARF_LIBRARY)
            if (NOT ELF_LIBRARY)
                message(STATUS "Please install the elfutils-libelf package")
            endif (NOT ELF_LIBRARY)
        else (FEDORA OR REDHAT)
            if (NOT DWARF_INCLUDE_DIR)
                message(STATUS "Could NOT find dwarf include dir")
            endif (NOT DWARF_INCLUDE_DIR)
            if (NOT LIBDW_INCLUDE_DIR)
                message(STATUS "Could NOT find libdw include dir")
            endif (NOT LIBDW_INCLUDE_DIR)
            if (NOT DWARF_LIBRARY)
                message(STATUS "Could NOT find libdw library")
            endif (NOT DWARF_LIBRARY)
            if (NOT ELF_LIBRARY)
                message(STATUS "Could NOT find libelf library")
            endif (NOT ELF_LIBRARY)
        endif (FEDORA OR REDHAT)
        message(FATAL_ERROR "Could NOT find some ELF and DWARF libraries, please install the missing packages")
    endif (DWARF_FIND_REQUIRED)
endif (DWARF_FOUND)

mark_as_advanced(DWARF_INCLUDE_DIR LIBDW_INCLUDE_DIR DWARF_LIBRARY ELF_LIBRARY)
include_directories(${DWARF_INCLUDE_DIR} ${LIBDW_INCLUDE_DIR})

message(STATUS "Checking availability of DWARF and ELF development libraries - done")