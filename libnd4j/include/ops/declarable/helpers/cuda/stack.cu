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
// Created by Yurii Shyrma on 02.01.2018
//

#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>


namespace nd4j {
namespace ops {
namespace helpers {


	///////////////////////////////////////////////////////////////////
	template <typename T>
	static void stack_(const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim) {

	}

	void stack(graph::LaunchContext* context, const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr.dataType(), stack_, (inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim), LIBND4J_TYPES);

}
}
}

