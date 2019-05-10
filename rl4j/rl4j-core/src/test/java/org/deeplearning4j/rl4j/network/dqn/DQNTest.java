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

package org.deeplearning4j.rl4j.network.dqn;

import org.junit.Test;
import org.nd4j.linalg.learning.config.RmsProp;

import java.io.File;
import java.io.IOException;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author saudet
 */
public class DQNTest {

    public static DQNFactoryStdDense.Configuration NET_CONF =
            new DQNFactoryStdDense.Configuration(
                    3,         //number of layers
                    16,        //number of hidden nodes
                    0.001,     //l2 regularization
                    new RmsProp(0.0005), null
            );

    @Test
    public void testModelLoadSave() throws IOException {
        DQN dqn = new DQNFactoryStdDense(NET_CONF).buildDQN(new int[] {42}, 13);

        File file = File.createTempFile("rl4j-dqn-", ".model");
        dqn.save(file.getAbsolutePath());

        DQN dqn2 = DQN.load(file.getAbsolutePath());

        assertEquals(dqn.mln, dqn2.mln);
    }
}
