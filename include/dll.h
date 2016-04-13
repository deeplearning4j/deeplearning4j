//
// Created by agibsonccc on 3/5/16.
//

#ifndef NATIVEOPERATIONS_DLL_H
#define NATIVEOPERATIONS_DLL_H
#ifdef _WIN32
//#include <windows.h>
#  define ND4J_EXPORT __declspec(dllexport)
#else
#  define ND4J_EXPORT
#endif
#endif //NATIVEOPERATIONS_DLL_H
