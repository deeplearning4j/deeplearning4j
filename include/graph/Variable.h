//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLE_H
#define LIBND4J_VARIABLE_H


#include <NDArray.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class Variable {
        protected:
            int32_t _id;
            nd4j::NDArray<T> * _ndarray;

            bool _external;
            bool _readOnly;

        public:
            Variable(nd4j::NDArray<T> *array);

            Variable(const nd4j::graph::FlatVariable *flatVariable);
            ~Variable();


            nd4j::NDArray<T> *getNDArray();
            bool isExternal();
            bool isReadOnly();

            void markExternal(bool reallyExternal);
            void markReadOnly(bool reallyReadOnly);
        };
    }
}

template <typename T>
bool nd4j::graph::Variable<T>::isExternal() {
    return _external;
}

template <typename T>
bool nd4j::graph::Variable<T>::isReadOnly() {
    return _readOnly;
}

template <typename T>
void nd4j::graph::Variable<T>::markExternal(bool reallyExternal) {
    this->_external = reallyExternal;
}

template <typename T>
void nd4j::graph::Variable<T>::markReadOnly(bool reallyReadOnly) {
   this->_readOnly = reallyReadOnly;
}

template <typename T>
nd4j::NDArray<T> * nd4j::graph::Variable<T>::getNDArray() {
    return this->_ndarray;
}

template <typename T>
nd4j::graph::Variable<T>::Variable(const nd4j::graph::FlatVariable *flatVariable) {
    int shapeLen = flatVariable->shape()->Length();
    int *shape = new int[shapeLen];
    this->_id = flatVariable->id();

    _external = true;
    _readOnly = false;

#pragma omp simd
    for (int e = 0; e < shapeLen; e++) {
        shape[e] = flatVariable->shape()->Get(e);
    }

    int bufLen = flatVariable->values()->Length();
    T *buffer = new T[bufLen];

#pragma omp simd
    for (int e = 0; e < bufLen; e++) {
        buffer[e] = (T) flatVariable->values()->Get(e);
    }

    _ndarray = new NDArray<T>(buffer, shape);
    _ndarray->_allocated = true;
}

template <typename T>
nd4j::graph::Variable<T>::Variable(NDArray<T> *array) {
    _ndarray = array;

    _external = false;
    _readOnly = false;
    _id = 0;
}

template <typename T>
nd4j::graph::Variable<T>::~Variable() {
    if (_ndarray != nullptr)
        delete _ndarray;
}

#endif //LIBND4J_VARIABLE_H
