#include <helpers/cpu/loops/IndexReductionLoops.hpp>
#include <types/types.h>
#include <system/type_boilerplate.h>

// First instantiate the base class
#define LIST_CALLBACK_BASE(INPUT) \
    template class sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>;

ITERATE_LIST((SD_COMMON_TYPES), LIST_CALLBACK_BASE)

// Now instantiate the operations with int32_t index type
#define LIST_CALLBACK_OPS_int32_t(INPUT) \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexMin<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::LastIndex<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::FirstIndex<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexMax<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*);

ITERATE_LIST((SD_COMMON_TYPES), LIST_CALLBACK_OPS_int32_t)

// And with long long index type
#define LIST_CALLBACK_OPS_LONG(INPUT) \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexMin<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::LastIndex<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::FirstIndex<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexMax<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*);

ITERATE_LIST((SD_COMMON_TYPES), LIST_CALLBACK_OPS_LONG)

#undef LIST_CALLBACK_BASE
#undef LIST_CALLBACK_OPS_int32_t
#undef LIST_CALLBACK_OPS_LONG