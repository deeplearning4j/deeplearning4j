//
// Created by agibsonccc on 3/5/16.
//

#ifndef NATIVEOPERATIONS_POINTERCAST_H
#define NATIVEOPERATIONS_POINTERCAST_H
#ifdef __APPLE__
#define Nd4jPointer long long
#endif
#ifdef _WIN32
#define Nd4jPointer long long
#endif

#ifdef __linux__
#define Nd4jPointer long
#endif

#endif //NATIVEOPERATIONS_POINTERCAST_H

//__declspec(dllexport)

