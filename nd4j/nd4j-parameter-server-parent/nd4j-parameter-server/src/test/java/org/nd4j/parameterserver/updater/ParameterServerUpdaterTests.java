/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.parameterserver.updater;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.aeron.ipc.NDArrayMessage;
import org.nd4j.aeron.ndarrayholder.InMemoryNDArrayHolder;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.updater.storage.NoUpdateStorage;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.*;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class ParameterServerUpdaterTests extends BaseND4JTest {

    @Test()
    @Timeout(30000L)
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
        assertNotNull(updater.toJson());


    }

}
