//
// Created by agibsonccc on 3/31/24.
//
#include <ops/declarable/helpers/addBias.h>
#include <helpers/ShapeUtils.h>

namespace sd {
namespace ops {
//////////////////////////////////////////////////////////////////////////
namespace helpers {

template <typename X>
SD_INLINE void addBiasBp_(graph::Context& block,
                          const NDArray* input,
                          const NDArray* gradO,
                          NDArray* gradI,
                          NDArray* gradB)  {

  const bool isNCHW = !block.getBArguments()->empty() ? B_ARG(0) : false;
  const int channelDim = isNCHW ? 1 : input->rankOf() - 1;  // second or last

  gradI->assign(gradO);

  std::vector<LongType> channel;
  channel.push_back(channelDim);
  auto dims = ShapeUtils::evalDimsToExclude(gradO->rankOf(), 1,channel.data());
  gradO->reduceAlongDimension(reduce::Sum, *gradB, dims);
  delete dims;
}


void addBiasBp(graph::Context& block,
               const NDArray* input,
               const NDArray* gradO,
               NDArray* gradI,
               NDArray* gradB)  {
  BUILD_SINGLE_SELECTOR(input->dataType(), addBiasBp_, (block,input, gradO, gradI, gradB), SD_FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void addBiasBp_, (graph::Context& block, const NDArray* input,
    const NDArray* gradO,
    NDArray* gradI, NDArray* gradB), SD_FLOAT_TYPES);

}
}
}