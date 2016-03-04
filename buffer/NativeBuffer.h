//
// Created by agibsonccc on 2/26/16.
//

#ifndef NATIVEOPERATIONS_NATIVEBUFFER_H
#define NATIVEOPERATIONS_NATIVEBUFFER_H
#include <stdlib.h>
class JavaCppDoublePointer {
private:
    double *buffer = NULL;

public:
    void create(int length);

    ~JavaCppDoublePointer();

    long long bufferAddress();

    void putDouble(int i,double vla);

    double *bufferRef();

};

class JavaCppFloatPointer {
private:
    float *buffer = NULL;
public:
    void create(int length);
    ~JavaCppFloatPointer();
    long long bufferAddress();
    void putFloat(int i,float val);
    float *bufferRef();
};

class JavaCppIntPointer {
private:
    int *buffer = NULL;
public:
    void create(int length);

    ~JavaCppIntPointer();

    long long bufferAddress();

    void putInt(int i,int val);

    int *bufferRef();
};


#endif //NATIVEOPERATIONS_NATIVEBUFFER_H
