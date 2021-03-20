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

package org.deeplearning4j.rl4j.mdp.gym;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 *
 * @author saudet
 */
@Tag(TagNames.FILE_IO)
@NativeTag
public class GymEnvTest {

    @Test
    @Disabled("Permissions issues on CI")
    public void testCartpole() {
        GymEnv mdp = new GymEnv("CartPole-v0", false, false);
        assertArrayEquals(new int[] {4}, mdp.getObservationSpace().getShape());
        assertEquals(2, mdp.getActionSpace().getSize());
        assertEquals(false, mdp.isDone());
        Box o = (Box)mdp.reset();
        StepReply r = mdp.step(0);
        assertEquals(4, o.getData().shape()[0]);
        assertEquals(4, ((Box)r.getObservation()).getData().shape()[0]);
        assertNotEquals(null, mdp.newInstance());
        mdp.close();
    }
}
