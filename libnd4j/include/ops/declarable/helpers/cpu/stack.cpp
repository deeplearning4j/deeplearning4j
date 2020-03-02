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
#include <execution/Threads.h>


namespace sd {
namespace ops {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
static void stack_(const std::vector<const NDArray*>& inArrs, NDArray* outArr, const int dim) {

	if(inArrs[0]->rankOf() == 0) {
	    int inSize = inArrs.size();

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++)
                outArr->p<T>(i, inArrs[i]->t<T>(0));
        };

        samediff::Threads::parallel_for(func, 0, inSize);
	}
	else {

		std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(outArr->rankOf(), {dim});
		auto list = outArr->allTensorsAlongDimension(dimsToExclude);		// list.size() == block.width()
        int listSize = list.size();

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++)
                list.at(i)->assign(inArrs[i]);
        };
        samediff::Threads::parallel_tad(func, 0, listSize);
	}
}

	void stack(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray* outArr, const int dim) {
		BUILD_SINGLE_SELECTOR(outArr->dataType(), stack_, (inArrs, outArr, dim), LIBND4J_TYPES);
	}

	BUILD_SINGLE_TEMPLATE(template void stack_ , (const std::vector<const NDArray*>& inArrs, NDArray* outArr, const int dim), LIBND4J_TYPES);

}
}
}

