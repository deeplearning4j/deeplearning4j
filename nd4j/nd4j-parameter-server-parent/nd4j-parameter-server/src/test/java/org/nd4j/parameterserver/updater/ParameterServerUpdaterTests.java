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

package org.nd4j.parameterserver.updater;

import org.junit.Test;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.aeron.ndarrayholder.InMemoryNDArrayHolder;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.updater.storage.NoUpdateStorage;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 12/2/16.
 */
public class ParameterServerUpdaterTests {

    @Test(timeout = 30000L)
    public void synchronousTest() {
        int cores = Runtime.getRuntime().availableProcessors();
        ParameterServerUpdater updater = new SynchronousParameterUpdater(new NoUpdateStorage(),
                        new InMemoryNDArrayHolder(Nd4j.zeros(2, 2)), cores);
        for (int i = 0; i < cores; i++) {
            updater.update(NDArrayMessage.wholeArrayUpdate(Nd4j.ones(2, 2)));
        }

        assertTrue(updater.shouldReplicate());
        updater.reset();
        assertFalse(updater.shouldReplicate());
        assumeNotNull(updater.toJson());


    }

}
