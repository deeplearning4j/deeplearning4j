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

package org.nd4j.common.io;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;

import static org.junit.Assert.assertEquals;

public class ClassPathResourceTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Test
    public void testDirExtractingIntelliJ() throws Exception {
        //https://github.com/deeplearning4j/deeplearning4j/issues/6483

        ClassPathResource cpr = new ClassPathResource("somedir");

        File f = testDir.newFolder();

        cpr.copyDirectory(f);

        File[] files = f.listFiles();
        assertEquals(1, files.length);
        assertEquals("afile.txt", files[0].getName());
    }

}
