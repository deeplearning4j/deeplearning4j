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
void stack(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& outArr, const int dim) {

	if(inArrs[0]->rankOf() == 0) {

#pragma omp parallel for if(inArrs.size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
		for(int i=0; i < inArrs.size(); ++i)
			outArr(i) = (*inArrs[i])(0.);
	}
	else {

		std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(outArr.rankOf(), {dim});	
		ResultSet<T>* list = outArr.allTensorsAlongDimension(dimsToExclude);		// list.size() == block.width()
		
#pragma omp parallel for if(list->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
		for(int i=0; i<list->size(); ++i)
			list->at(i)->assign(inArrs[i]);
		
		delete list;
	}
}


template void stack<float>  (const std::vector<NDArray<float  >*>& inArrs, NDArray<float  >& outArr, const int dim);
template void stack<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& outArr, const int dim);
template void stack<double> (const std::vector<NDArray<double >*>& inArrs, NDArray<double >& outArr, const int dim);


}
}
}

