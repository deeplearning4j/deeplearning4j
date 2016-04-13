# Need to find both Qt{4,5} and QGLViewer if the QQL support is to be built
FIND_PACKAGE(Qt4 COMPONENTS QtCore QtXml QtOpenGL QtGui)
IF(NOT Qt4_FOUND)
	FIND_PACKAGE(Qt5 QUIET COMPONENTS Core Xml OpenGL Gui Widgets)
	IF(NOT Qt4_FOUND AND NOT Qt5_FOUND)
		MESSAGE("Qt{4,5} not found. Install it and set Qt{4,5}_DIR accordingly")
		IF (WIN32)
			MESSAGE("  In Windows, Qt5_DIR should be something like C:/Qt/5.4/msvc2013_64_opengl/lib/cmake/Qt5")
		ENDIF()
	ENDIF()
ENDIF()

FIND_PATH(QGLVIEWER_INCLUDE_DIR qglviewer.h
    /usr/include/QGLViewer
    /opt/local/include/QGLViewer
    /usr/local/include/QGLViewer
    /sw/include/QGLViewer
    ENV QGLVIEWERROOT
  )

find_library(QGLVIEWER_LIBRARY_RELEASE
  NAMES qglviewer-qt4 qglviewer QGLViewer QGLViewer2
  PATHS /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
        ENV QGLVIEWERROOT
        ENV LD_LIBRARY_PATH
        ENV LIBRARY_PATH
  PATH_SUFFIXES QGLViewer QGLViewer/release
)
find_library(QGLVIEWER_LIBRARY_DEBUG
  NAMES dqglviewer dQGLViewer dQGLViewer2 QGLViewerd2
  PATHS /usr/lib
        /usr/local/lib
        /opt/local/lib
        /sw/lib
        ENV QGLVIEWERROOT
        ENV LD_LIBRARY_PATH
        ENV LIBRARY_PATH
  PATH_SUFFIXES QGLViewer QGLViewer/release
)

if(QGLVIEWER_LIBRARY_RELEASE)
  if(QGLVIEWER_LIBRARY_DEBUG)
    set(QGLVIEWER_LIBRARY optimized ${QGLVIEWER_LIBRARY_RELEASE} debug ${QGLVIEWER_LIBRARY_DEBUG})
  else()
    set(QGLVIEWER_LIBRARY ${QGLVIEWER_LIBRARY_RELEASE})
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QGLVIEWER DEFAULT_MSG
  QGLVIEWER_INCLUDE_DIR QGLVIEWER_LIBRARY)
