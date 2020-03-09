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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/stack.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>


namespace sd {
namespace ops {
namespace helpers {


///////////////////////////////////////////////////////////////////
template <typename T>
static void stack_(const std::vector<const NDArray*>& inArrs, NDArray& output, const int dim) {

	const int numOfSubArrs = inArrs.size();

	if(inArrs[0]->rankOf() == 0) {

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++)
                output.p<T>(i, inArrs[i]->t<T>(0));
        };

        sd::Threads::parallel_for(func, 0, numOfSubArrs);
	}
	else {

		auto zTadPack = ConstantTadHelper::getInstance()->tadForDimensions(output.getShapeInfo(), ShapeUtils::evalDimsToExclude(output.rankOf(), {dim}));
		Nd4jLong* zTadShapeInfo  = zTadPack.primaryShapeInfo();

        auto func = PRAGMA_THREADS_FOR {

            for (auto i = start; i < stop; i++) {

                void* zBuff = output.bufferWithOffset(zTadPack.primaryOffsets()[i]);

                NativeOpExecutioner::execTransformAny(inArrs[0]->getContext(), transform::Assign,
                                                     inArrs[i]->getBuffer(), inArrs[i]->getShapeInfo(), nullptr/*input specialBuffer*/,  nullptr/*input specialShapeInfo*/,
                                                     zBuff,                  zTadShapeInfo,             nullptr/*output specialBuffer*/, nullptr/*output specialShapeInfo*/,
                                                     nullptr, nullptr, nullptr, false/*allowParallelism*/);
            }
        };

        sd::Threads::parallel_tad(func, 0, numOfSubArrs);
    }

}

////////////////////////////////////////////////////////////////////////
void stack(sd::LaunchContext * context, const std::vector<const NDArray*>& inArrs, NDArray& output, const int dim) {
	BUILD_SINGLE_SELECTOR(output.dataType(), stack_, (inArrs, output, dim), LIBND4J_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void stack_ , (const std::vector<const NDArray*>& inArrs, NDArray& output, const int dim), LIBND4J_TYPES);


///////////////////////////////////////////////////////////////////
template <typename T>
static void unstack_(const NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {

	const int numOfSubArrs = outArrs.size();

	if(outArrs[0]->rankOf() == 0) {

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++)
                outArrs[i]->p<T>(0, input.t<T>(i));
        };

        sd::Threads::parallel_for(func, 0, numOfSubArrs);
	}
	else {

		auto xTadPack = ConstantTadHelper::getInstance()->tadForDimensions(input.getShapeInfo(), ShapeUtils::evalDimsToExclude(input.rankOf(), {dim}));
		Nd4jLong* xTadShapeInfo  = xTadPack.primaryShapeInfo();

        auto func = PRAGMA_THREADS_FOR {
            for (auto i = start; i < stop; i++) {

                void* xBuff = input.bufferWithOffset(xTadPack.primaryOffsets()[i]);

                NativeOpExecutioner::execTransformAny(input.getContext(), transform::Assign,
                									 xBuff,                   xTadShapeInfo,              nullptr/*input specialBuffer*/, nullptr/*input specialShapeInfo*/,
                                                     outArrs[i]->getBuffer(), outArrs[i]->getShapeInfo(), nullptr/*output specialBuffer*/,  nullptr/*output specialShapeInfo*/,
                                                     nullptr, nullptr, nullptr, false/*allowParallelism*/);
            }
        };

        sd::Threads::parallel_tad(func, 0, numOfSubArrs);
	}
}

////////////////////////////////////////////////////////////////////////
void unstack(sd::LaunchContext* context, const NDArray& input, const std::vector<NDArray*>& outArrs, const int dim) {
	BUILD_SINGLE_SELECTOR(input.dataType(), unstack_, (input, outArrs, dim), LIBND4J_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void unstack_, (const NDArray& input, const std::vector<NDArray*>& outArrs, const int dim), LIBND4J_TYPES);

}
}
}

