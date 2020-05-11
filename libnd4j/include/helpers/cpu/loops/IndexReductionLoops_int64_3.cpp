/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include "./IndexReductionLoops.hpp"

BUILD_DOUBLE_TEMPLATE(template void sd::IndexReductionLoops, ::wrapIndexReduce(const int opNum, const void* vx, const Nd4jLong* xShapeInfo, void* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, void* vextraParams), LIBND4J_TYPES_3, (sd::DataType::INT64, Nd4jLong));