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
    void extractPatches(NDArray<T>* images, NDArray<T>* output, int sizeRow, int sizeCol, int stradeRow, int stradeCol, int rateRow, int rateCol, bool theSame);

}
}
}
#endif
