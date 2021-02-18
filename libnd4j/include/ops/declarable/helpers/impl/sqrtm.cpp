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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include <ops/declarable/CustomOperations.h>
#include <helpers/Sqrtm.h>

namespace sd 	  {
namespace ops 	  {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void sqrtm_(const NDArray* x, NDArray* z) {


    if(x->rankOf() == 2) {

        ops::helpers::Sqrtm<T>::calc(*x, *z);
    }
    else {

        auto listX = x->allTensorsAlongDimension({-2, -1});
        auto listZ = z->allTensorsAlongDimension({-2, -1});

        auto func = PRAGMA_THREADS_FOR {

            for (auto i = start; i < stop; i++)
                ops::helpers::Sqrtm<T>::calc(*listX.at(i), *listZ.at(i));
        };

        samediff::Threads::parallel_tad(func, 0, listX.size());
    }
}


//////////////////////////////////////////////////////////////////////////
void sqrtm(sd::LaunchContext* context, const NDArray* x, NDArray* z) {

    x->syncToHost();
    BUILD_SINGLE_SELECTOR(z->dataType(), sqrtm_, (x, z), FLOAT_TYPES);
    z->syncToDevice();
}



}
}
}
