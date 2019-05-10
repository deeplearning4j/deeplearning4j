/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
#include <dll.h>

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
