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
// Created by raver119 on 20/04/18.
//

#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/parity_ops.h>
#include <helpers/DebugInfo.h>
#include <execution/Threads.h>

namespace sd {
    DebugInfo DebugHelper::debugStatistics(NDArray const* input) {
        DebugInfo info;
        DebugHelper::retrieveDebugStatistics(&info, input);
        return info;
    }
    void 
    DebugHelper::retrieveDebugStatistics(DebugInfo* info, NDArray const* input) {
        if (nullptr == info)
                return;
        
        info->_minValue = 0.;
        info->_maxValue = -1;
        info->_meanValue = 0.;
        info->_stdDevValue = 1.;
        info->_zeroCount = 0;
        info->_positiveCount = 0;
        info->_negativeCount = 0;
        info->_infCount = 0;
        info->_nanCount = 0;
        if (input->lengthOf() == 1) { // scalar case
            info->_minValue = input->e<double>(0);
            info->_maxValue = info->_minValue;
            info->_meanValue = info->_minValue;
            info->_stdDevValue = info->_minValue;
            info->_zeroCount = sd::math::nd4j_abs(input->e<double>(0)) > 0.00001? 0: 1;
            info->_positiveCount = input->e<double>(0) > 0?1:0;
            info->_negativeCount = input->e<double>(0) < 0?1:0;
            info->_infCount = sd::math::nd4j_isinf(input->e<double>(0));
            info->_nanCount = sd::math::nd4j_isnan(input->e<double>(0));
        }
        else if (input->lengthOf() > 0) {
            // TO DO: here processing for all elements with array
            auto _minValue = input->e<double>(0);
            auto _maxValue = input->e<double>(0);
            auto _meanValue = input->e<double>(0);
            auto _stdDevValue = 0.; //info->_minValue;
            auto _zeroCount = sd::math::nd4j_abs(input->e<double>(0)) > 0.00001? 0L : 1L;
            auto _positiveCount = input->e<double>(0) > 0? 1L : 0L;
            auto _negativeCount = input->e<double>(0) < 0? 1L : 0L;
            auto _infCount = sd::math::nd4j_isinf(input->e<double>(0)) ? 1L : 0L;
            auto _nanCount = sd::math::nd4j_isnan(input->e<double>(0)) ? 1L : 0L;

PRAGMA_OMP_PARALLEL_FOR_ARGS(schedule(guided) reduction(+:_nanCount,_infCount,_meanValue,_zeroCount,_positiveCount,_negativeCount) reduction(min:_minValue) reduction(max:_maxValue))
            for (Nd4jLong e = 1; e < input->lengthOf(); e++) {
                auto current = input->e<double>(e);
                auto n = e + 1.;
//                auto delta = current - _meanValue;
//                auto delta2 = delta * delta;
                _minValue = sd::math::nd4j_min(current, _minValue);
                _maxValue = sd::math::nd4j_max(current, _maxValue);

                _meanValue += current;
                //_meanValue += delta / n; // this is a perfect formula but not working with omp in this notation
                //_stdDevValue += delta2 * e / n;

                _zeroCount += sd::math::nd4j_abs(current) > 0.00001 ? 0 : 1;
                _positiveCount += current > 0 ? 1 : 0;
                _negativeCount += current < 0 ? 1 : 0;
                _infCount += sd::math::nd4j_isinf(current);
                _nanCount += sd::math::nd4j_isnan(current);
            }
            *info = {_minValue, _maxValue, _meanValue / input->lengthOf(), _stdDevValue, _zeroCount, _positiveCount, _negativeCount, _infCount, _nanCount};
            _stdDevValue = 0; //math::nd4j_sqrt<double, double>(info->_stdDevValue / (input->lengthOf() - 1));

            auto func = PRAGMA_REDUCE_DOUBLE {
                auto _stdDevValue = 0.0;
                for (auto e = start; e < stop; e++) {
                    double current = input->e<double>(e);
                    _stdDevValue += (info->_meanValue - current) * (info->_meanValue - current); //info->_minValue;
                }

                return _stdDevValue;
            };
            _stdDevValue = samediff::Threads::parallel_double(func, LAMBDA_AD { return _old + _new; }, 0, input->lengthOf());

            info->_stdDevValue = math::nd4j_sqrt<double, double>(_stdDevValue / input->lengthOf());

        }
// else - no statistics for empty
    }
}
