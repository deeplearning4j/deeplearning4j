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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_OPS_DECLARABLE_COMMON_H
#define LIBND4J_OPS_DECLARABLE_COMMON_H

#include <memory>
#include <op_boilerplate.h>
#include <types/float16.h>
#include <NDArray.h>
#include <graph/Context.h>
#include <ops/declarable/DeclarableOp.h>
#include <ops/declarable/BooleanOp.h>
#include <ops/declarable/LogicOp.h>
#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableCustomOp.h>
#include <ops/declarable/DeclarableListOp.h>
#include <ops/declarable/OpRegistrator.h>
#include <helpers/ArrayUtils.h>
#include <helpers/ShapeUtils.h>
#include <array/ShapeList.h>
#include <Status.h>
#include <stdexcept>

#endif