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

	if(inArrs[0]->rankOf() == 0) {

#pragma omp parallel for if(inArrs.size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
		for(int i=0; i < inArrs.size(); ++i)
			outArr.putScalar(i, inArrs[i]->getScalar<T>(0));
	}
	else {

		std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(outArr.rankOf(), {dim});
		auto list = outArr.allTensorsAlongDimension(dimsToExclude);		// list.size() == block.width()
		
#pragma omp parallel for if(list->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
		for(int i=0; i<list->size(); ++i)
			list->at(i)->assign(inArrs[i]);
		
		delete list;
	}
}

	void stack(const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr.dataType(), stack_, (inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (const std::vector<NDArray*>& inArrs, NDArray& outArr, const int dim), LIBND4J_TYPES);

}
}
}

