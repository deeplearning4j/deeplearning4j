#include <helpers/cpu/loops/IndexReductionLoops.hpp>
#include <types/types.h>
#include <system/type_boilerplate.h>

// First instantiate the base class
#define LIST_CALLBACK_BASE(INPUT) \
    template class sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>;

ITERATE_LIST((SD_COMMON_TYPES), LIST_CALLBACK_BASE)

// Now instantiate the operations with int index type
#define LIST_CALLBACK_OPS_INT(INPUT) \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int>::loopIndexReduce<simdOps::IndexMin<GET_SECOND(INPUT), int>>(GET_SECOND(INPUT)*, const long long*, int*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int>::loopIndexReduce<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), int>>(GET_SECOND(INPUT)*, const long long*, int*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int>::loopIndexReduce<simdOps::LastIndex<GET_SECOND(INPUT), int>>(GET_SECOND(INPUT)*, const long long*, int*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int>::loopIndexReduce<simdOps::FirstIndex<GET_SECOND(INPUT), int>>(GET_SECOND(INPUT)*, const long long*, int*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int>::loopIndexReduce<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), int>>(GET_SECOND(INPUT)*, const long long*, int*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int>::loopIndexReduce<simdOps::IndexMax<GET_SECOND(INPUT), int>>(GET_SECOND(INPUT)*, const long long*, int*, const long long*, const long long*, const long long*, void*);

ITERATE_LIST((SD_COMMON_TYPES), LIST_CALLBACK_OPS_INT)

// And with long long index type
#define LIST_CALLBACK_OPS_LONG(INPUT) \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexMin<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const long long*, sd::LongType*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const long long*, sd::LongType*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::LastIndex<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const long long*, sd::LongType*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::FirstIndex<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const long long*, sd::LongType*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const long long*, sd::LongType*, const long long*, const long long*, const long long*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexMax<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const long long*, sd::LongType*, const long long*, const long long*, const long long*, void*);

ITERATE_LIST((SD_COMMON_TYPES), LIST_CALLBACK_OPS_LONG)

#undef LIST_CALLBACK_BASE
#undef LIST_CALLBACK_OPS_INT
#undef LIST_CALLBACK_OPS_LONG