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
package org.deeplearning4j.parallelism;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.parallel.MultiBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

@DisplayName("Multi Boolean Test")
@NativeTag
@Tag(TagNames.DL4J_OLD_API)
class MultiBooleanTest extends BaseDL4JTest {

    @Test
    @DisplayName("Test Boolean 1")
    void testBoolean1() throws Exception {
        MultiBoolean bool = new MultiBoolean(5);
        assertTrue(bool.allFalse());
        assertFalse(bool.allTrue());
    }

    @Test
    @DisplayName("Test Boolean 2")
    void testBoolean2() throws Exception {
        MultiBoolean bool = new MultiBoolean(5);
        bool.set(true, 2);
        assertFalse(bool.allFalse());
        assertFalse(bool.allTrue());
    }

    @Test
    @DisplayName("Test Boolean 3")
    void testBoolean3() throws Exception {
        MultiBoolean bool = new MultiBoolean(5);
        bool.set(true, 0);
        bool.set(true, 1);
        bool.set(true, 2);
        bool.set(true, 3);
        assertFalse(bool.allTrue());
        bool.set(true, 4);
        assertFalse(bool.allFalse());
        assertTrue(bool.allTrue());
        bool.set(false, 2);
        assertFalse(bool.allTrue());
        bool.set(true, 2);
        assertTrue(bool.allTrue());
    }

    @Test
    @DisplayName("Test Boolean 4")
    void testBoolean4() throws Exception {
        MultiBoolean bool = new MultiBoolean(5, true);
        assertTrue(bool.get(1));
        bool.set(false, 1);
        assertFalse(bool.get(1));
    }

    @Test
    @DisplayName("Test Boolean 5")
    void testBoolean5() throws Exception {
        MultiBoolean bool = new MultiBoolean(5, true, true);
        for (int i = 0; i < 5; i++) {
            bool.set(false, i);
        }
        for (int i = 0; i < 5; i++) {
            bool.set(true, i);
        }
        assertTrue(bool.allFalse());
    }
}
