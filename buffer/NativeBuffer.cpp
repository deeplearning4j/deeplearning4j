//
// Created by agibsonccc on 2/26/16.
//
#include "NativeBuffer.h"


void JavaCppDoublePointer::create(int length) {
    this->buffer = (double *) malloc(sizeof(double) * length);
}

JavaCppDoublePointer::~JavaCppDoublePointer() {
    if(this->buffer != NULL)
        free(this->buffer);
}

long JavaCppDoublePointer::bufferAddress() {
    return reinterpret_cast<long>(this->buffer);
}

void JavaCppDoublePointer::putDouble(int i, double vla) {
    this->buffer[i] = vla;
}


double * JavaCppDoublePointer::bufferRef() {
    return this->buffer;
}

void JavaCppFloatPointer::putFloat(int i, float val) {
    this->buffer[i] = val;
}


void JavaCppFloatPointer::create(int length) {
    this->buffer = (float *) malloc(sizeof(float) * length);
}
JavaCppFloatPointer::~JavaCppFloatPointer() {
    if(this->buffer != NULL)
        free(this->buffer);
}

long JavaCppFloatPointer::bufferAddress() {
    return reinterpret_cast<long>(this->buffer);
}

float * JavaCppFloatPointer::bufferRef() {
    return this->buffer;
}





void JavaCppIntPointer::create(int length) {
    this->buffer = (int *) malloc(sizeof(int) * length);
}

JavaCppIntPointer::~JavaCppIntPointer(){
    if(this->buffer != NULL)
        free(this->buffer);
}

void JavaCppIntPointer::putInt(int i, int val) {
    this->buffer[i] = val;
}


long JavaCppIntPointer::bufferAddress() {
    return reinterpret_cast<long>(this->buffer);
}

int * JavaCppIntPointer::bufferRef() {
    return this->buffer;
}
