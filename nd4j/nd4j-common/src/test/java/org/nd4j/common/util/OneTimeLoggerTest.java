/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.util;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.common.util.OneTimeLogger;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class OneTimeLoggerTest {

    @Test
    public void testLogger1() throws Exception {
        OneTimeLogger.info(log, "Format: {}; Pew: {};", 1, 2);
    }

    @Test
    public void testBuffer1() throws Exception {
        assertTrue(OneTimeLogger.isEligible("Message here"));

        assertFalse(OneTimeLogger.isEligible("Message here"));

        assertTrue(OneTimeLogger.isEligible("Message here 23"));
    }
}
