find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_IEEE80211 gnuradio-ieee80211)

FIND_PATH(
    GR_IEEE80211_INCLUDE_DIRS
    NAMES gnuradio/ieee80211/api.h
    HINTS $ENV{IEEE80211_DIR}/include
        ${PC_IEEE80211_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_IEEE80211_LIBRARIES
    NAMES gnuradio-ieee80211
    HINTS $ENV{IEEE80211_DIR}/lib
        ${PC_IEEE80211_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-ieee80211Target.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_IEEE80211 DEFAULT_MSG GR_IEEE80211_LIBRARIES GR_IEEE80211_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_IEEE80211_LIBRARIES GR_IEEE80211_INCLUDE_DIRS)
