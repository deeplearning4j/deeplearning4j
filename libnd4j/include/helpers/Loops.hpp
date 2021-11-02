/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/DataTypeUtils.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/shape.h>

namespace sd {}
// template void Loops::loopReduce<double, double>(const double* x, const sd::LongType* tadShapeInfo, const
// sd::LongType* tadOffsets, double* z, const sd::LongType* zShapeInfo, double* extraParams, std::function<double(const
// double*)> startVal, std::function<double(double,double,double*)> update, std::function<double(double,double*)> op,
// std::function<double(double,sd::LongType,double*)> postPr); template void Loops::loopReduce<float, float>(const float*
// x, const sd::LongType* tadShapeInfo, const sd::LongType* tadOffsets, float* z, const sd::LongType* zShapeInfo, float*
// extraParams, std::function<float(const float*)> startVal, std::function<float(float,float,float*)> update,
// std::function<float(float,float*)> op, std::function<float(float,sd::LongType,float*)> postPr);

//#endif // LIBND4J_LOOPS_CPP
