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
package org.datavec.image;

import org.apache.commons.io.FileUtils;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.io.ClassPathResource;
import java.io.File;
import java.util.Arrays;
import java.util.List;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.junit.jupiter.api.DisplayName;
import java.nio.file.Path;
import org.junit.jupiter.api.extension.ExtendWith;

@DisplayName("Label Generator Test")
class LabelGeneratorTest {


    @Test
    @DisplayName("Test Parent Path Label Generator")
    @Disabled
    void testParentPathLabelGenerator(@TempDir Path testDir) throws Exception {
        File orig = new ClassPathResource("datavec-data-image/testimages/class0/0.jpg").getFile();
        for (String dirPrefix : new String[] { "m.", "m" }) {
            File f = testDir.toFile();
            int numDirs = 3;
            int filesPerDir = 4;
            for (int i = 0; i < numDirs; i++) {
                File currentLabelDir = new File(f, dirPrefix + i);
                currentLabelDir.mkdirs();
                for (int j = 0; j < filesPerDir; j++) {
                    File f3 = new File(currentLabelDir, "myImg_" + j + ".jpg");
                    FileUtils.copyFile(orig, f3);
                    assertTrue(f3.exists());
                }
            }
            ImageRecordReader rr = new ImageRecordReader(28, 28, 1, new ParentPathLabelGenerator());
            rr.initialize(new FileSplit(f));
            List<String> labelsAct = rr.getLabels();
            List<String> labelsExp = Arrays.asList(dirPrefix + "0", dirPrefix + "1", dirPrefix + "2");
            assertEquals(labelsExp, labelsAct);
            int expCount = numDirs * filesPerDir;
            int actCount = 0;
            while (rr.hasNext()) {
                rr.next();
                actCount++;
            }
            assertEquals(expCount, actCount);
        }
    }
}
