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
// @author George A. Shulinok <sgazeos@gmail.com>
//

#ifndef __LIBND4J_L_GAMMA__H__
#define __LIBND4J_L_GAMMA__H__

#include <ops/declarable/helpers/helpers.h>
#include "array/NDArray.h"

namespace sd {
namespace ops {
namespace helpers {

    // calculate the digamma function for each element for array
    void lgamma(sd::LaunchContext* context, NDArray& x, NDArray& z);

}
}
}


#endif //__LIBND4J_L_GAMMA__H__
