//
// Created by agibsonccc on 8/30/24.
//

#ifndef LIBND4J_RESHAPENOCOPY_H
#define LIBND4J_RESHAPENOCOPY_H
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {
bool reshapeNoAlloc(const LongType* inShape,
                    const std::vector<sd::LongType>& newShape,
                    char order,
                    sd::LongType* outShape);
}
}
}
#endif  // LIBND4J_RESHAPENOCOPY_H
