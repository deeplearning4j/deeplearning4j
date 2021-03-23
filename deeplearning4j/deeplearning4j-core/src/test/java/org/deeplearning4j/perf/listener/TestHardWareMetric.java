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

package org.deeplearning4j.perf.listener;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.core.listener.HardwareMetric;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import oshi.json.SystemInfo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

@Disabled("AB 2019/05/24 - Failing on CI - \"Could not initialize class oshi.jna.platform.linux.Libc\" - Issue #7657")
@NativeTag
@Tag(TagNames.JACKSON_SERDE)
public class TestHardWareMetric extends BaseDL4JTest {

    @Test
    public void testHardwareMetric() {
        HardwareMetric hardwareMetric = HardwareMetric.fromSystem(new SystemInfo());
        assertNotNull(hardwareMetric);
        System.out.println(hardwareMetric);

        String yaml = hardwareMetric.toYaml();
        HardwareMetric fromYaml = HardwareMetric.fromYaml(yaml);
        assertEquals(hardwareMetric, fromYaml);
    }

}
