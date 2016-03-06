//
// Created by agibsonccc on 3/5/16.
//

#ifndef NATIVEOPERATIONS_DLL_H
#define NATIVEOPERATIONS_DLL_H
#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#elseif
#    define LIBRARY_API
#endif
#endif //NATIVEOPERATIONS_DLL_H
