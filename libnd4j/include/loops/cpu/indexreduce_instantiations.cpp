#include <helpers/cpu/loops/IndexReductionLoops.hpp>
#include <types/types.h>
#include <system/type_boilerplate.h>
#include "indexreduce.hpp"

// Macro to conditionally instantiate based on HAS_* flags
#define CONDITIONAL_INSTANTIATE(TYPE_TUPLE, MACRO_CALL) \
    CONDITIONAL_INSTANTIATE_IMPL(GET_SECOND(TYPE_TUPLE), MACRO_CALL)

#define CONDITIONAL_INSTANTIATE_IMPL(TYPE, MACRO_CALL) \
    CONDITIONAL_INSTANTIATE_##TYPE(MACRO_CALL)

// Define conditional instantiation for each type
#ifdef HAS_FLOAT32
#define CONDITIONAL_INSTANTIATE_float(MACRO_CALL) MACRO_CALL((FLOAT32, float))
#else
#define CONDITIONAL_INSTANTIATE_float(MACRO_CALL)
#endif

#ifdef HAS_DOUBLE
#define CONDITIONAL_INSTANTIATE_double(MACRO_CALL) MACRO_CALL((DOUBLE, double))
#else
#define CONDITIONAL_INSTANTIATE_double(MACRO_CALL)
#endif

#ifdef HAS_FLOAT16
#define CONDITIONAL_INSTANTIATE_float16(MACRO_CALL) MACRO_CALL((HALF, float16))
#else
#define CONDITIONAL_INSTANTIATE_float16(MACRO_CALL)
#endif

#ifdef HAS_BFLOAT16
#define CONDITIONAL_INSTANTIATE_bfloat16(MACRO_CALL) MACRO_CALL((BFLOAT16, bfloat16))
#else
#define CONDITIONAL_INSTANTIATE_bfloat16(MACRO_CALL)
#endif

#ifdef HAS_INT8
#define CONDITIONAL_INSTANTIATE_int8_t(MACRO_CALL) MACRO_CALL((INT8, int8_t))
#else
#define CONDITIONAL_INSTANTIATE_int8_t(MACRO_CALL)
#endif

#ifdef HAS_INT16
#define CONDITIONAL_INSTANTIATE_int16_t(MACRO_CALL) MACRO_CALL((INT16, int16_t))
#else
#define CONDITIONAL_INSTANTIATE_int16_t(MACRO_CALL)
#endif

#ifdef HAS_INT32
#define CONDITIONAL_INSTANTIATE_Int32Type(MACRO_CALL) MACRO_CALL((INT32, Int32Type))
#else
#define CONDITIONAL_INSTANTIATE_Int32Type(MACRO_CALL)
#endif

#ifdef HAS_LONG
#define CONDITIONAL_INSTANTIATE_LongType(MACRO_CALL) MACRO_CALL((INT64, LongType))
#else
#define CONDITIONAL_INSTANTIATE_LongType(MACRO_CALL)
#endif

#ifdef HAS_UINT8
#define CONDITIONAL_INSTANTIATE_uint8_t(MACRO_CALL) MACRO_CALL((UINT8, uint8_t))
#else
#define CONDITIONAL_INSTANTIATE_uint8_t(MACRO_CALL)
#endif

#ifdef HAS_UINT16
#define CONDITIONAL_INSTANTIATE_uint16_t(MACRO_CALL) MACRO_CALL((UINT16, uint16_t))
#else
#define CONDITIONAL_INSTANTIATE_uint16_t(MACRO_CALL)
#endif

#ifdef HAS_UINT32
#define CONDITIONAL_INSTANTIATE_uint32_t(MACRO_CALL) MACRO_CALL((UINT32, uint32_t))
#else
#define CONDITIONAL_INSTANTIATE_uint32_t(MACRO_CALL)
#endif

#ifdef HAS_UNSIGNEDLONG
#define CONDITIONAL_INSTANTIATE_uint64_t(MACRO_CALL) MACRO_CALL((UINT64, uint64_t))
#else
#define CONDITIONAL_INSTANTIATE_uint64_t(MACRO_CALL)
#endif

#ifdef HAS_BOOL
#define CONDITIONAL_INSTANTIATE_bool(MACRO_CALL) MACRO_CALL((BOOL, bool))
#else
#define CONDITIONAL_INSTANTIATE_bool(MACRO_CALL)
#endif

// Now instantiate the operations with int32_t index type (only if HAS_INT32 is defined)
#ifdef HAS_INT32

#define LIST_CALLBACK_OPS_int32_t(INPUT) \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexMin<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::LastIndex<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::FirstIndex<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), int32_t>::loopIndexReduce<simdOps::IndexMax<GET_SECOND(INPUT), int32_t>>(GET_SECOND(INPUT)*, const sd::LongType*, int32_t*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*);

// Conditionally instantiate for each type
#ifdef HAS_FLOAT32
LIST_CALLBACK_OPS_int32_t((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
LIST_CALLBACK_OPS_int32_t((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
LIST_CALLBACK_OPS_int32_t((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
LIST_CALLBACK_OPS_int32_t((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
LIST_CALLBACK_OPS_int32_t((INT8, int8_t))
#endif
#ifdef HAS_INT16
LIST_CALLBACK_OPS_int32_t((INT16, int16_t))
#endif
#ifdef HAS_INT32
LIST_CALLBACK_OPS_int32_t((INT32, Int32Type))
#endif
#ifdef HAS_LONG
LIST_CALLBACK_OPS_int32_t((INT64, LongType))
#endif
#ifdef HAS_UINT8
LIST_CALLBACK_OPS_int32_t((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
LIST_CALLBACK_OPS_int32_t((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
LIST_CALLBACK_OPS_int32_t((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
LIST_CALLBACK_OPS_int32_t((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
LIST_CALLBACK_OPS_int32_t((BOOL, bool))
#endif

#endif // HAS_INT32

// And with long long index type (only if HAS_LONG is defined)
#ifdef HAS_LONG

#define LIST_CALLBACK_OPS_LONG(INPUT) \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexMin<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::LastIndex<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::FirstIndex<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*); \
    template void sd::IndexReductionLoops<GET_SECOND(INPUT), sd::LongType>::loopIndexReduce<simdOps::IndexMax<GET_SECOND(INPUT), sd::LongType>>(GET_SECOND(INPUT)*, const sd::LongType*, sd::LongType*, const sd::LongType*, const sd::LongType*, const sd::LongType*, void*);

// Conditionally instantiate for each type
#ifdef HAS_FLOAT32
LIST_CALLBACK_OPS_LONG((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
LIST_CALLBACK_OPS_LONG((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
LIST_CALLBACK_OPS_LONG((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
LIST_CALLBACK_OPS_LONG((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
LIST_CALLBACK_OPS_LONG((INT8, int8_t))
#endif
#ifdef HAS_INT16
LIST_CALLBACK_OPS_LONG((INT16, int16_t))
#endif
#ifdef HAS_INT32
LIST_CALLBACK_OPS_LONG((INT32, Int32Type))
#endif
#ifdef HAS_LONG
LIST_CALLBACK_OPS_LONG((INT64, LongType))
#endif
#ifdef HAS_UINT8
LIST_CALLBACK_OPS_LONG((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
LIST_CALLBACK_OPS_LONG((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
LIST_CALLBACK_OPS_LONG((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
LIST_CALLBACK_OPS_LONG((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
LIST_CALLBACK_OPS_LONG((BOOL, bool))
#endif

#endif // HAS_LONG

// Instantiate execScalar methods with int32_t
#ifdef HAS_INT32

#define INSTANTIATE_EXEC_SCALAR_INT32(INPUT) \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar<simdOps::IndexMin<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar<simdOps::LastIndex<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar<simdOps::FirstIndex<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar<simdOps::IndexMax<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*);

#ifdef HAS_FLOAT32
INSTANTIATE_EXEC_SCALAR_INT32((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_EXEC_SCALAR_INT32((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_EXEC_SCALAR_INT32((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_EXEC_SCALAR_INT32((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_EXEC_SCALAR_INT32((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_EXEC_SCALAR_INT32((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_EXEC_SCALAR_INT32((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_EXEC_SCALAR_INT32((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_EXEC_SCALAR_INT32((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_EXEC_SCALAR_INT32((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_EXEC_SCALAR_INT32((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_EXEC_SCALAR_INT32((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_EXEC_SCALAR_INT32((BOOL, bool))
#endif

#endif // HAS_INT32

// Instantiate execScalar methods with sd::LongType
#ifdef HAS_LONG

#define INSTANTIATE_EXEC_SCALAR_LONG(INPUT) \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar<simdOps::IndexMin<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar<simdOps::LastIndex<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar<simdOps::FirstIndex<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*); \
    template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar<simdOps::IndexMax<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*);

#ifdef HAS_FLOAT32
INSTANTIATE_EXEC_SCALAR_LONG((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_EXEC_SCALAR_LONG((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_EXEC_SCALAR_LONG((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_EXEC_SCALAR_LONG((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_EXEC_SCALAR_LONG((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_EXEC_SCALAR_LONG((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_EXEC_SCALAR_LONG((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_EXEC_SCALAR_LONG((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_EXEC_SCALAR_LONG((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_EXEC_SCALAR_LONG((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_EXEC_SCALAR_LONG((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_EXEC_SCALAR_LONG((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_EXEC_SCALAR_LONG((BOOL, bool))
#endif

#endif // HAS_LONG

// Instantiate exec methods with int32_t
#ifdef HAS_INT32

#define INSTANTIATE_EXEC_INT32(INPUT) \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec<simdOps::IndexMin<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec<simdOps::LastIndex<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec<simdOps::FirstIndex<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec<simdOps::IndexMax<GET_SECOND(INPUT), int32_t>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*);

#ifdef HAS_FLOAT32
INSTANTIATE_EXEC_INT32((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_EXEC_INT32((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_EXEC_INT32((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_EXEC_INT32((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_EXEC_INT32((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_EXEC_INT32((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_EXEC_INT32((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_EXEC_INT32((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_EXEC_INT32((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_EXEC_INT32((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_EXEC_INT32((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_EXEC_INT32((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_EXEC_INT32((BOOL, bool))
#endif

#endif // HAS_INT32

// Instantiate exec methods with sd::LongType
#ifdef HAS_LONG

#define INSTANTIATE_EXEC_LONG(INPUT) \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec<simdOps::IndexMin<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec<simdOps::IndexAbsoluteMin<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec<simdOps::LastIndex<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec<simdOps::FirstIndex<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec<simdOps::IndexAbsoluteMax<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*); \
    template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec<simdOps::IndexMax<GET_SECOND(INPUT), sd::LongType>>(const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*);

#ifdef HAS_FLOAT32
INSTANTIATE_EXEC_LONG((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_EXEC_LONG((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_EXEC_LONG((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_EXEC_LONG((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_EXEC_LONG((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_EXEC_LONG((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_EXEC_LONG((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_EXEC_LONG((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_EXEC_LONG((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_EXEC_LONG((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_EXEC_LONG((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_EXEC_LONG((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_EXEC_LONG((BOOL, bool))
#endif

#endif // HAS_LONG

// Add these instantiations for the non-templated methods (the ones with int opNum parameter)
#ifdef HAS_INT32

#define INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32(INPUT) \
   template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::execScalar(int, const void*, const sd::LongType*, void*);

#ifdef HAS_FLOAT32
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_INT32((BOOL, bool))
#endif

#endif // HAS_INT32

#ifdef HAS_LONG

#define INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG(INPUT) \
   template sd::LongType functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::execScalar(int, const void*, const sd::LongType*, void*);

#ifdef HAS_FLOAT32
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_NON_TEMPLATED_EXEC_SCALAR_LONG((BOOL, bool))
#endif

#endif // HAS_LONG

#ifdef HAS_INT32

#define INSTANTIATE_NON_TEMPLATED_EXEC_INT32(INPUT) \
   template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), int32_t>::exec(int, const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*);

#ifdef HAS_FLOAT32
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_NON_TEMPLATED_EXEC_INT32((BOOL, bool))
#endif

#endif // HAS_INT32

#ifdef HAS_LONG

#define INSTANTIATE_NON_TEMPLATED_EXEC_LONG(INPUT) \
   template void functions::indexreduce::IndexReduce<GET_SECOND(INPUT), sd::LongType>::exec(int, const void*, const sd::LongType*, void*, void*, const sd::LongType*, sd::LongType*, sd::LongType, const sd::LongType*, const sd::LongType*);

#ifdef HAS_FLOAT32
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((FLOAT32, float))
#endif
#ifdef HAS_DOUBLE
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((DOUBLE, double))
#endif
#ifdef HAS_FLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((HALF, float16))
#endif
#ifdef HAS_BFLOAT16
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((BFLOAT16, bfloat16))
#endif
#ifdef HAS_INT8
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((INT8, int8_t))
#endif
#ifdef HAS_INT16
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((INT16, int16_t))
#endif
#ifdef HAS_INT32
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((INT32, Int32Type))
#endif
#ifdef HAS_LONG
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((INT64, LongType))
#endif
#ifdef HAS_UINT8
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((UINT8, uint8_t))
#endif
#ifdef HAS_UINT16
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((UINT16, uint16_t))
#endif
#ifdef HAS_UINT32
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((UINT32, uint32_t))
#endif
#ifdef HAS_UNSIGNEDLONG
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((UINT64, uint64_t))
#endif
#ifdef HAS_BOOL
INSTANTIATE_NON_TEMPLATED_EXEC_LONG((BOOL, bool))
#endif

#endif // HAS_LONG