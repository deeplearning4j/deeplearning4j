//
// Created by agibsonccc on 2/6/16.
//

#ifndef NATIVEOPERATIONS_CBLAS_ENUM_CONVERSION_H_H
#define NATIVEOPERATIONS_CBLAS_ENUM_CONVERSION_H_H
#include <cblas.h>

/*
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
    AtlasConj=114};
enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG  {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};
*/

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Converts a character
 * to its proper enum
 * for row (c) or column (f) ordering
 * default is row major
 */
CBLAS_ORDER convertOrder(int from);
/**
 * Converts a character to its proper enum
 * t -> transpose
 * n -> no transpose
 * c -> conj
 */
CBLAS_TRANSPOSE convertTranspose(int from);
/**
 * Upper or lower
 * U/u -> upper
 * L/l -> lower
 *
 * Default is upper
 */
CBLAS_UPLO convertUplo(int from);

/**
 * For diagonals:
 * u/U -> unit
 * n/N -> non unit
 *
 * Default: unit
 */
CBLAS_DIAG convertDiag(int from);
/**
 * Side of a matrix, left or right
 * l /L -> left
 * r/R -> right
 * default: left
 */
CBLAS_SIDE convertSide(int from);

#ifdef __cplusplus
}
#endif


#endif //NATIVEOPERATIONS_CBLAS_ENUM_CONVERSION_H_H
