//
// Created by agibsonccc on 10/15/24.
//
#include <system/type_boilerplate.h>
#include <types/types.h>
#include <loops/pairwise_transform.h>
#include "pairwise.hpp"

//Note this file is here
// because instantiations in header
//files can cause out of memory in the compiler.
//we technically need only 1 instance of the instantiations.


/*
 *
 *
ITERATE_COMBINATIONS_3: This macro iterates over three lists of data types (SD_COMMON_TYPES) and applies the INSTANT_PROCESS_COMBINATION_3 macro to each combination. This results in the instantiation of the PairWiseTransform::exec function for each combination of data types.
ITERATE_COMBINATIONS: This macro iterates over two lists of data types (SD_COMMON_TYPES) and applies the CALLBACK_INSTANTIATE_PROMOTE macro to each combination. This is likely used for promoting data types.
Function Instantiation:
The PairWiseTransform::exec function is instantiated for various combinations of data types. The function signature includes parameters for operation number (opNum), input arrays (x, y), their shape information (xShapeInfo, yShapeInfo), output array (z), its shape information (zShapeInfo), extra parameters (extraParams), and the range of elements to process (start, stop).
 * */
ITERATE_COMBINATIONS_3((SD_NUMERIC_TYPES),(SD_NUMERIC_TYPES),(SD_NUMERIC_TYPES),INSTANT_PROCESS_COMBINATION_3,functions::pairwise_transforms::PairWiseTransform, ::exec(int opNum, const void *x, const sd::LongType *xShapeInfo, const void *y,
                                                                                                                                                                         const sd::LongType *yShapeInfo, void *z, const sd::LongType *zShapeInfo,
                                                                                                                                                                         void *extraParams, sd::LongType start, sd::LongType stop))
ITERATE_COMBINATIONS(
    (SD_NUMERIC_TYPES),
    (SD_NUMERIC_TYPES),
    CALLBACK_INSTANTIATE_PROMOTE,
    promote,
    ;
)