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

package org.deeplearning4j.optimize.solvers.accumulation;


import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.concurrent.BlockingQueue;

/**
 * This class provides additional functionality to FancyBlockingQueue: it tracks memory use of stored compressed INDArrays, and if their size becomes too big, it:
 * a) decompresses them into single INDArray
 * b) removes original updates messages
 * c) keeps updating single INDArray until it gets consumed
 * d) once that happened - it automatically switches back to original behavior
 *
 * @author raver119@gmail.com
 */
public class SmartFancyBlockingQueue extends FancyBlockingQueue<INDArray> {

    public SmartFancyBlockingQueue(BlockingQueue<INDArray> queue) {
        super(queue);
    }
}
