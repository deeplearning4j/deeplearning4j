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

package org.nd4j.parameterserver.distributed.logic;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.Timeout;
import org.nd4j.parameterserver.distributed.logic.completion.FrameCompletionHandler;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
@Ignore
@Deprecated
public class FrameCompletionHandlerTest {
    @Before
    public void setUp() throws Exception {

    }

    @Rule
    public Timeout globalTimeout = Timeout.seconds(30);

    /**
     * This test emulates 2 frames being processed at the same time
     * @throws Exception
     */
    @Test
    public void testCompletion1() throws Exception {
        FrameCompletionHandler handler = new FrameCompletionHandler();
        long[] frames = new long[] {15L, 17L};
        long[] originators = new long[] {123L, 183L};
        for (Long originator : originators) {
            for (Long frame : frames) {
                for (int e = 1; e <= 512; e++) {
                    handler.addHook(originator, frame, (long) e);
                }
            }

            for (Long frame : frames) {
                for (int e = 1; e <= 512; e++) {
                    handler.notifyFrame(originator, frame, (long) e);
                }
            }
        }


        for (Long originator : originators) {
            for (Long frame : frames) {
                assertEquals(true, handler.isCompleted(originator, frame));
            }
        }
    }

}
