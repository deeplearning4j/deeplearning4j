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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
// 
//
// @author AbdelRauf    (rauf@konduit.ai)
//

#ifndef LIBND4J_HELPERS_IMAGES_H
#define LIBND4J_HELPERS_IMAGES_H

#include <system/op_boilerplate.h>
#include <math/templatemath.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

    void transformRgbGrs(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC);

    void transformHsvRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC);

    void transformRgbHsv(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC);
    void transformYuvRgb(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC);
    void transformRgbYuv(sd::LaunchContext* context, const NDArray& input, NDArray& output, const int dimC);

    void transformYiqRgb(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC);

    void transformRgbYiq(sd::LaunchContext* context, const NDArray* input, NDArray* output, const int dimC);
}
}
}

#endif