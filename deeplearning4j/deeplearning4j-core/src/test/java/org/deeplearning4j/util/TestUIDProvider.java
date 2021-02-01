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

package org.deeplearning4j.util;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.core.util.UIDProvider;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Created by Alex on 26/06/2016.
 */
public class TestUIDProvider extends BaseDL4JTest {

    @Test
    public void testUIDProvider() {
        String jvmUID = org.deeplearning4j.core.util.UIDProvider.getJVMUID();
        String hardwareUID = org.deeplearning4j.core.util.UIDProvider.getHardwareUID();

        assertNotNull(jvmUID);
        assertNotNull(hardwareUID);

        assertTrue(!jvmUID.isEmpty());
        assertTrue(!hardwareUID.isEmpty());

        assertEquals(jvmUID, org.deeplearning4j.core.util.UIDProvider.getJVMUID());
        assertEquals(hardwareUID, UIDProvider.getHardwareUID());

        System.out.println("JVM uid:      " + jvmUID);
        System.out.println("Hardware uid: " + hardwareUID);
    }

}
