//
//  @author sgazeos@gmail.com
//
#ifndef __AXIS_H_HELPERS__
#define __AXIS_H_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void adjustAxis(NDArray<T>* input, NDArray<T>* axisVector, std::vector<int>& output);

}
}
}
#endif
