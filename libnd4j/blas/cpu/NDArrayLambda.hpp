


template<typename T>
void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<T(T, T, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if (second == nullptr) {
        nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Second is NULL\n","");
        throw std::runtime_error("second is null");
    }

    if (third == nullptr) {
        nd4j_printf("applyTriplewiseLambda requires three operands to be valid NDArrays, but Third is NULL\n","");
        throw std::runtime_error("third is null");
    }
    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != second->dataType() || dataType() != third->dataType() || dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyTriplewiseLambda<T> method: bother four arrays (this, second, third, target) should have the same type !");

    if (this->lengthOf() != second->lengthOf() || this->lengthOf() != third->lengthOf() || !this->isSameShape(second) || !this->isSameShape(third)) {
        nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
        throw std::runtime_error("Shapes mismach");
    }

    auto f = this->bufferAsT<T>();
    auto s = second->bufferAsT<T>();
    auto t = third->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == second->ordering() && this->ordering() == third->ordering()  && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == second->ews() && this->ews() == third->ews()) {

        auto loop = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment)
                z[e] = func(f[e], s[e], t[e]);
        };

        samediff::Threads::parallel_for(loop, 0, _length);
    } else {
        if (f == z) {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto tOffset = this->getOffset(e);
                    auto uOffset = second->getOffset(e);
                    auto vOffset = third->getOffset(e);

                    f[tOffset] = func(f[tOffset], s[uOffset], t[vOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        } else {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto tOffset = this->getOffset(e);
                    auto uOffset = second->getOffset(e);
                    auto vOffset = third->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(f[tOffset], s[uOffset], t[vOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        }
    }
}
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<double (double, double, double)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float (float, float, float)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<float16 (float16, float16, float16)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bfloat16 (bfloat16, bfloat16, bfloat16)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int (int, int, int)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int16_t (int16_t, int16_t, int16_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint8_t (uint8_t, uint8_t, uint8_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint16_t (uint16_t, uint16_t, uint16_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint32_t (uint32_t, uint32_t, uint32_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<uint64_t (uint64_t, uint64_t, uint64_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<int8_t (int8_t, int8_t, int8_t)>& func, NDArray* target);
template void NDArray::applyTriplewiseLambda(NDArray* second, NDArray *third, const std::function<bool (bool, bool, bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<T(T, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if (other == nullptr) {
        nd4j_printf("applyPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
        throw std::runtime_error("Other is null");
    }

    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != other->dataType() || dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyPairwiseLambda<T> method: all three arrays (this, other, target) must have the same type !");

    if (this->lengthOf() != other->lengthOf()) {
        nd4j_printf("applyPairwiseLambda requires both operands to have the same shape\n","");
        throw std::runtime_error("Shapes mismach");
    }

    auto f = this->bufferAsT<T>();
    auto s = other->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {

        auto loop = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment)
                z[e] = func(f[e], s[e]);
        };

        samediff::Threads::parallel_for(loop, 0, _length);
    } else {
        if (f == z) {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);

                    f[xOffset] = func(f[xOffset], s[yOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        } else {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(f[xOffset], s[yOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        }
    }
}
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<double (double, double)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<float (float, float)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<float16 (float16, float16)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<bfloat16 (bfloat16, bfloat16)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int (int, int)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int16_t (int16_t, int16_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint8_t (uint8_t, uint8_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint16_t (uint16_t, uint16_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint32_t (uint32_t, uint32_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<uint64_t (uint64_t, uint64_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<int8_t (int8_t, int8_t)>& func, NDArray* target);
template void NDArray::applyPairwiseLambda(const NDArray* other, const std::function<bool (bool, bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyLambda(const std::function<T(T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyLambda<T> method: types of this and target array should match !");

    auto f = this->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {

        auto loop = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment)
                z[e] = func(f[e]);
        };

        samediff::Threads::parallel_for(loop, 0, _length);
    } else {
        if (f == z) {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);

                    f[xOffset] = func(f[xOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        } else {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(f[xOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        }
    }
}
template void NDArray::applyLambda(const std::function<double(double)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<float(float)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<float16(float16)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<bfloat16(bfloat16)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<Nd4jLong(Nd4jLong)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<int16_t(int16_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<int32_t(int32_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<uint8_t(uint8_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<uint16_t(uint16_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<uint32_t(uint32_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<uint64_t(uint64_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<int8_t(int8_t)>& func, NDArray* target);
template void NDArray::applyLambda(const std::function<bool(bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyIndexedLambda(const std::function<T(Nd4jLong, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyIndexedLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyIndexedLambda<T> method: types of this and target array should match !");

    auto f = this->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1)) {

        auto loop = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment)
                z[e] = func(e, f[e]);
        };

        samediff::Threads::parallel_for(loop, 0, _length);
    } else {
        if (f == z) {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);

                    f[xOffset] = func(e, f[xOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        } else {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func(e, f[xOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        }
    }
}
template void NDArray::applyIndexedLambda(const std::function<double(Nd4jLong, double)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<float(Nd4jLong, float)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<float16(Nd4jLong, float16)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<bfloat16(Nd4jLong, bfloat16)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<Nd4jLong(Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<int(Nd4jLong, int)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<int16_t(Nd4jLong, int16_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<uint8_t (Nd4jLong, uint8_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<uint16_t (Nd4jLong, uint16_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<uint32_t (Nd4jLong, uint32_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<uint64_t (Nd4jLong, uint64_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<int8_t(Nd4jLong, int8_t)>& func, NDArray* target);
template void NDArray::applyIndexedLambda(const std::function<bool(Nd4jLong, bool)>& func, NDArray* target);

//////////////////////////////////////////////////////////////////////////
template<typename T>
void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<T(Nd4jLong, T, T)>& func, NDArray* target) {
    if (target == nullptr)
        target = this;

    if (other == nullptr) {
        nd4j_printf("applyIndexedPairwiseLambda requires both operands to be valid NDArrays, but Y is NULL\n","");
        throw std::runtime_error("Other is null");
    }
    if(dataType() != DataTypeUtils::fromT<T>())
        throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: wrong template parameter T, its type should be the same as type of this array!");
    if(dataType() != target->dataType())
        throw std::runtime_error("NDArray::applyIndexedPairwiseLambda<T> method: types of this and target array should match !");
    if (this->lengthOf() != other->lengthOf()) {
        nd4j_printf("applyIndexedPairwiseLambda requires both operands to have the same shape\n","");
        throw std::runtime_error("Shapes mismach");
    }

    auto f = this->bufferAsT<T>();
    auto s = other->bufferAsT<T>();
    auto z = target->bufferAsT<T>();

    if (this->ordering() == other->ordering() && this->ordering() == target->ordering() && (this->ews() == 1 && target->ews() == 1) && this->ews() == other->ews()) {

        auto loop = PRAGMA_THREADS_FOR {
            for (auto e = start; e < stop; e += increment)
                z[e] = func((Nd4jLong) e, f[e], s[e]);
        };

        samediff::Threads::parallel_for(loop, 0, _length);
    } else {
        if (f == z) {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);

                    f[xOffset] = func((Nd4jLong) e, f[xOffset], s[yOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        } else {

            auto loop = PRAGMA_THREADS_FOR {
                for (auto e = start; e < stop; e += increment) {
                    auto xOffset = this->getOffset(e);
                    auto yOffset = other->getOffset(e);
                    auto zOffset = target->getOffset(e);

                    z[zOffset] = func((Nd4jLong) e, f[xOffset], s[yOffset]);
                }
            };

            samediff::Threads::parallel_for(loop, 0, _length);
        }
    }
}
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<double (Nd4jLong, double, double)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float (Nd4jLong, float, float)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<float16 (Nd4jLong, float16, float16)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bfloat16 (Nd4jLong, bfloat16, bfloat16)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<Nd4jLong (Nd4jLong, Nd4jLong, Nd4jLong)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int (Nd4jLong, int, int)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int16_t (Nd4jLong, int16_t, int16_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint8_t (Nd4jLong, uint8_t, uint8_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint16_t (Nd4jLong, uint16_t, uint16_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint32_t (Nd4jLong, uint32_t, uint32_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<uint64_t (Nd4jLong, uint64_t, uint64_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<int8_t (Nd4jLong, int8_t, int8_t)>& func, NDArray* target);
template void NDArray::applyIndexedPairwiseLambda(NDArray* other, const std::function<bool (Nd4jLong, bool, bool)>& func, NDArray* target);