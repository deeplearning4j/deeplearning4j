//
// @author raver119@gmail.com
//

#ifndef LIBND4J_VARIABLE_H
#define LIBND4J_VARIABLE_H

#include <string>
#include <NDArray.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class Variable {
        protected:
            int32_t _id;
            nd4j::NDArray<T> * _ndarray = nullptr;
            std::string _name;

            bool _external;
            bool _readOnly;
            bool _placeholder = false;

        public:
            Variable(bool placeHolder);
            Variable(nd4j::NDArray<T> *array = nullptr, const char *name = nullptr);
            Variable(const nd4j::graph::FlatVariable *flatVariable);
            ~Variable();


            nd4j::NDArray<T> *getNDArray();
            void setNDArray(nd4j::NDArray<T> * array);
            bool isExternal();
            bool isReadOnly();
            bool isEmpty();

            bool isPlaceholder();
            bool hasNDArray();

            void markExternal(bool reallyExternal);
            void markReadOnly(bool reallyReadOnly);

            int32_t id();
            void setId(int32_t id);

            std::string *getName();
            void setName(std::string *name);
        };
    }
}

template <typename T>
bool nd4j::graph::Variable<T>::hasNDArray() {
    return _ndarray != nullptr;
}

template <typename T>
bool nd4j::graph::Variable<T>::isPlaceholder() {
    return _placeholder;
}

template <typename T>
std::string * nd4j::graph::Variable<T>::getName() {
    return &_name;
}

template <typename T>
void nd4j::graph::Variable<T>::setName(std::string *name) {
    _name = *name;
}

template <typename T>
int32_t nd4j::graph::Variable<T>::id() {
    return _id;
}

template <typename T>
void nd4j::graph::Variable<T>::setId(int32_t id) {
    _id = id;
}

template <typename T>
bool nd4j::graph::Variable<T>::isEmpty() {
    return _ndarray == nullptr || !_ndarray->nonNull();
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
void nd4j::graph::Variable<T>::setNDArray(nd4j::NDArray<T> * array) {
    this->_ndarray = array;
}

template <typename T>
nd4j::graph::Variable<T>::Variable(const nd4j::graph::FlatVariable *flatVariable) {
    int shapeLen = flatVariable->shape()->Length();
    int *shape = new int[shapeLen];
    this->_id = flatVariable->id();

    if (flatVariable->name() != nullptr && flatVariable->name()->size() != 0)
        this->_name = flatVariable->name()->str();

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
    _ndarray->triggerAllocationFlag(true, true);
}

template <typename T>
nd4j::graph::Variable<T>::Variable(bool placeholder) {
    _placeholder = placeholder;
}

template <typename T>
nd4j::graph::Variable<T>::Variable(NDArray<T> *array, const char *name ) {
    _ndarray = array;

    _external = false;
    _readOnly = false;
    _id = 0;
    
    if (name != nullptr)
        _name = std::string(name);
}

template <typename T>
nd4j::graph::Variable<T>::~Variable() {
    if (_ndarray != nullptr)
        delete _ndarray;
}

#endif //LIBND4J_VARIABLE_H
