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

package org.nd4j.parameterserver.status.play;

import org.junit.Test;
import org.nd4j.parameterserver.model.SubscriberState;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 12/1/16.
 */
public class StorageTests {

    @Test(timeout = 20000L)
    public void testMapStorage() throws Exception {
        StatusStorage mapDb = new MapDbStatusStorage();
        assertEquals(SubscriberState.empty(), mapDb.getState(-1));


        SubscriberState noEmpty = SubscriberState.builder().isMaster(true).serverState("master").streamId(1).build();
        mapDb.updateState(noEmpty);
        assertEquals(noEmpty, mapDb.getState(1));

        Thread.sleep(10000);
        assertTrue(mapDb.numStates() == 0);

    }

    @Test(timeout = 20000L)
    public void testStorage() throws Exception {
        StatusStorage statusStorage = new InMemoryStatusStorage();
        assertEquals(SubscriberState.empty(), statusStorage.getState(-1));


        SubscriberState noEmpty = SubscriberState.builder().isMaster(true).serverState("master").streamId(1).build();
        statusStorage.updateState(noEmpty);
        assertEquals(noEmpty, statusStorage.getState(1));

        Thread.sleep(10000);
        assertTrue(statusStorage.numStates() == 0);

    }

}
