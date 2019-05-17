//
// Created by raver on 5/17/2019.
//

#include <array/ConstantHolder.h>
#include <DataTypeUtils.h>


namespace nd4j {
    bool ConstantHolder::hasBuffer(nd4j::DataType dataType) {
        return _buffers.count(dataType) > 0;
    }

    template <typename T>
    bool ConstantHolder::hasBuffer() {
        return hasBuffer(DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT bool ConstantHolder::hasBuffer, (), LIBND4J_TYPES);

    void ConstantHolder::addBuffer(ConstantDataBuffer &pointer, nd4j::DataType dataType) {
        _buffers[dataType] = pointer;
    }

    template <typename T>
    void ConstantHolder::addBuffer(ConstantDataBuffer &pointer) {
        addBuffer(pointer, DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT void ConstantHolder::addBuffer, (ConstantDataBuffer&), LIBND4J_TYPES);

    ConstantDataBuffer* ConstantHolder::getConstantDataBuffer(nd4j::DataType dataType) {
        if (!hasBuffer(dataType))
            throw std::runtime_error("Requested dataType is absent in storage");

        return &_buffers[dataType];
    }

    template <typename T>
    ConstantDataBuffer* ConstantHolder::getConstantDataBuffer() {
        return getConstantDataBuffer(DataTypeUtils::fromT<T>());
    }
    BUILD_SINGLE_TEMPLATE(template ND4J_EXPORT ConstantDataBuffer* ConstantHolder::getConstantDataBuffer, (), LIBND4J_TYPES);
}