//
// Created by agibsonccc on 3/10/25.
//

#ifndef LIBND4J_PAIRWISE_INSTANTIATIONS_SINGLE_H
#define LIBND4J_PAIRWISE_INSTANTIATIONS_SINGLE_H
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
// Single-type definitions for more granular CUDA template instantiations
//
#include <loops/pairwise_transform.h>
#include <system/type_boilerplate.h>
#include <types/types.h>

// Include the current instantiation definitions
#include "pairwise_instantiations.h"

// Define single-type groups for CUDA sharding using the SD_NUMERIC_TYPES
// Each SD_SINGLE_TYPE_N contains a single type from SD_NUMERIC_TYPES

// Float types (typically the first 4 types in SD_NUMERIC_TYPES)
#define SD_SINGLE_TYPE_0 (GET(0, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_1 (GET(1, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_2 (GET(2, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_3 (GET(3, (SD_NUMERIC_TYPES)))

// Integer types (typically the next 4 in SD_NUMERIC_TYPES)
#define SD_SINGLE_TYPE_4 (GET(4, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_5 (GET(5, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_6 (GET(6, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_7 (GET(7, (SD_NUMERIC_TYPES)))

// Additional types (remaining types in SD_NUMERIC_TYPES)
#define SD_SINGLE_TYPE_8 (GET(8, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_9 (GET(9, (SD_NUMERIC_TYPES)))
#define SD_SINGLE_TYPE_10 (GET(10, (SD_NUMERIC_TYPES)))

#endif  // LIBND4J_PAIRWISE_INSTANTIATIONS_SINGLE_H
