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
//  @author raver119@gmail.com
//

#include <graph/ExecutorConfiguration.h>

namespace nd4j {
    namespace graph {
        ExecutorConfiguration::ExecutorConfiguration(const nd4j::graph::FlatConfiguration *conf) {
            if (conf != nullptr) {
                _profilingMode = conf->profilingMode();
                _executionMode = conf->executionMode();
                _outputMode = conf->outputMode();
                _timestats = conf->timestats();
                _footprintForward = conf->footprintForward();
                _footprintBackward = conf->footprintBackward();
                _direction = conf->direction();
            } else {
                _profilingMode = ProfilingMode_NONE;
                _executionMode = ExecutionMode_SEQUENTIAL;
                _outputMode = OutputMode_IMPLICIT;
                _timestats = false;
            }
        };

        ExecutorConfiguration* ExecutorConfiguration::clone() {
            auto clone = new ExecutorConfiguration();
            clone->_profilingMode = _profilingMode;
            clone->_executionMode = _executionMode;
            clone->_outputMode = _outputMode;
            clone->_timestats = _timestats;
            clone->_direction = _direction;
            clone->_footprintForward = _footprintForward;
            clone->_footprintBackward = _footprintBackward;

            return clone;
        };
    }
}