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

package org.deeplearning4j.nn.conf;

import org.nd4j.linalg.learning.config.*;

/**
 *
 * All the possible different updaters
 *
 * @author Adam Gibson
 */
public enum Updater {
    SGD, ADAM, ADAMAX, ADADELTA, NESTEROVS, NADAM, ADAGRAD, RMSPROP, NONE, @Deprecated CUSTOM;


    public IUpdater getIUpdaterWithDefaultConfig() {
        switch (this) {
            case SGD:
                return new Sgd();
            case ADAM:
                return new Adam();
            case ADAMAX:
                return new AdaMax();
            case ADADELTA:
                return new AdaDelta();
            case NESTEROVS:
                return new Nesterovs();
            case NADAM:
                return new Nadam();
            case ADAGRAD:
                return new AdaGrad();
            case RMSPROP:
                return new RmsProp();
            case NONE:
                return new NoOp();
            case CUSTOM:
            default:
                throw new UnsupportedOperationException("Unknown or not supported updater: " + this);
        }
    }
}
