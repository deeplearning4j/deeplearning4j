//
// Created by raver on 2/28/2019.
//

#ifndef DEV_TESTS_OPEXECUTIONER_H
#define DEV_TESTS_OPEXECUTIONER_H

#include <NativeOpExcutioner.h>
#include <NDArray.h>

namespace nd4j {
    class ND4J_EXPORT OpBenchmark {
    protected:
        int _opNum;
        NDArray *_x;
        NDArray *_y;
        NDArray *_z;
        std::vector<int> _axis;
    public:
        OpBenchmark() = default;
        OpBenchmark(NDArray *x, NDArray *y, NDArray *z);
        OpBenchmark(NDArray *x, NDArray *z);
        OpBenchmark(NDArray *x, NDArray *z, std::initializer_list<int> axis);

        NDArray& x();
        int opNum();
        virtual void executeOnce() = 0;
    };
}


#endif //DEV_TESTS_OPEXECUTIONER_H
