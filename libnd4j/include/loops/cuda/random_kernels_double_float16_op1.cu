#include <loops/cuda/random_impl_gen.h>

DISPATCH_KERNEL_SIMPLE(randomDouble_, randomDoubleGeneric, float16,
                      INPUT(sd::Pointer state, void const* x, sd::LongType const* xShapeBuffer, void* z, sd::LongType const* zShapeBuffer, void* extraArguments),
                      PARAMS(state, x, xShapeBuffer, z, zShapeBuffer, extraArguments), OPS_A((1, DropOut)))
