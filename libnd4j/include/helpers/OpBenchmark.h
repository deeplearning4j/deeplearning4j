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
        NDArray *_x = nullptr;
        NDArray *_y = nullptr;
        NDArray *_z = nullptr;
        std::vector<int> _axis;
    public:
        OpBenchmark() = default;
        OpBenchmark(NDArray *x, NDArray *y, NDArray *z);
        OpBenchmark(NDArray *x, NDArray *z);
        OpBenchmark(NDArray *x, NDArray *z, std::initializer_list<int> axis);
        OpBenchmark(NDArray *x, NDArray *z, std::vector<int> axis);

        void setOpNum(int opNum);
        void setX(NDArray *array);
        void setY(NDArray *array);
        void setZ(NDArray *array);

        NDArray& x();
        int opNum();

        virtual std::string orders() = 0;

        virtual void executeOnce() = 0;

        virtual OpBenchmark* clone() = 0;
    };
}


#endif //DEV_TESTS_OPEXECUTIONER_H
