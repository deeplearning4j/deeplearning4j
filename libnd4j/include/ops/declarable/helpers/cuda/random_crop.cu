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
//  @author sgazeos@gmail.com
//

#include <ops/declarable/helpers/random_crop.h>
//#include <NativeOps.h>
#include <vector>
#include <memory>
#include <graph/Context.h>
namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static int _randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output, int seed) {
        return Status::OK();
    }

    int randomCropFunctor(graph::Context& context, NDArray* input, NDArray* shape, NDArray* output, int seed) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return _randomCropFunctor, (context, input, shape, output, seed), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template int _randomCropFunctor, (graph::Context& context, NDArray* input, NDArray* shape, NDArray* output,  int seed), FLOAT_TYPES);

}
}
}