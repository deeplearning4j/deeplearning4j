//
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {
    template <typename T>
    void FORCEINLINE _cross(NDArray<T> *a, NDArray<T> *b, NDArray<T> *o) {
        auto a0 = a->getScalar(0);
        auto a1 = a->getScalar(1);
        auto a2 = a->getScalar(2);

        auto b0 = b->getScalar(0);
        auto b1 = b->getScalar(1);
        auto b2 = b->getScalar(2);

        o->putScalar(0, a1 * b2 - a2 * b1);
        o->putScalar(1, a2 * b0 - a0 * b2);
        o->putScalar(2, a0 * b1 - a1 * b0);
    }

    template <typename T>
    void FORCEINLINE _crossBatched(NDArray<T> *a, NDArray<T> *b, NDArray<T> *o) {
        auto _a = a->reshape(a->ordering(), {-1, 3});
        auto _b = b->reshape(b->ordering(), {-1, 3});
        auto _o = o->reshape(o->ordering(), {-1, 3});

        auto tadsA = NDArrayFactory<T>::allTensorsAlongDimension(_a, {1});
        auto tadsB = NDArrayFactory<T>::allTensorsAlongDimension(_b, {1});
        auto tadsO = NDArrayFactory<T>::allTensorsAlongDimension(_o, {1});

        int tads = tadsA->size();

        #pragma omp parallel for simd schedule(static)
        for (int e = 0; e < tads; e++) {
            auto a_ = tadsA->at(e);
            auto b_ = tadsB->at(e);
            auto o_ = tadsO->at(e);

            helpers::_cross(a_, b_, o_);
        }

        delete tadsA;
        delete tadsB;
        delete tadsO;
        delete _a;
        delete _b;
        delete _o;
    }
}
}
}