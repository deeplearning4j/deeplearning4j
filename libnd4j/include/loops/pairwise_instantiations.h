//
// Created by agibsonccc on 11/22/24.
//

#ifndef LIBND4J_PAIRWISE_INSTANTIATIONS_H
#define LIBND4J_PAIRWISE_INSTANTIATIONS_H
/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
//
// Instantiated by genCompilation
//
#include <system/type_boilerplate.h>
#include <types/types.h>
#include <loops/pairwise_transform.h>
#include <loops/cpu/pairwise.hpp>

// Note: Instantiations are generated to prevent compiler memory issues

/*
 *
 * Function Instantiation:
 * PairWiseTransform::exec instantiated for types: @TYPE1@, @TYPE2@, @TYPE3@
 */



// Manually Defined Partitions Using GET Macros
#define SD_NUMERIC_TYPES_PART_0 (\
    GET(0, (SD_NUMERIC_TYPES)), \
    GET(1, (SD_NUMERIC_TYPES)), \
    GET(2, (SD_NUMERIC_TYPES)), \
    GET(3, (SD_NUMERIC_TYPES)), \
    GET(4, (SD_NUMERIC_TYPES)))

#define SD_NUMERIC_TYPES_PART_1 \
    (GET(5, (SD_NUMERIC_TYPES)), \
    GET(6, (SD_NUMERIC_TYPES)), \
    GET(7, (SD_NUMERIC_TYPES)), \
    GET(8, (SD_NUMERIC_TYPES))) \

#define SD_NUMERIC_TYPES_PART_2 \
 (GET(9, (SD_NUMERIC_TYPES)), \
     GET(10, (SD_NUMERIC_TYPES))) \


#endif  // LIBND4J_PAIRWISE_INSTANTIATIONS_H
