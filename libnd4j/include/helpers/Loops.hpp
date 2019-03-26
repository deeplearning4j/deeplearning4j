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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.03.2019
//

//#ifndef LIBND4J_LOOPS_CPP
//#define LIBND4J_LOOPS_CPP

#include <Loops.h>
#include <shape.h>
#include <OmpLaunchHelper.h>
#include <DataTypeUtils.h>


namespace nd4j {


}
//template void Loops::loopTadXZ<double, double>(const double* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, double* z, const Nd4jLong* zShapeInfo, double* extraParams, std::function<double(const double*)> startVal, std::function<double(double,double,double*)> update, std::function<double(double,double*)> op, std::function<double(double,Nd4jLong,double*)> postPr);
//template void Loops::loopTadXZ<float, float>(const float* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, float* z, const Nd4jLong* zShapeInfo, float* extraParams, std::function<float(const float*)> startVal, std::function<float(float,float,float*)> update, std::function<float(float,float*)> op, std::function<float(float,Nd4jLong,float*)> postPr);

//#endif // LIBND4J_LOOPS_CPP

