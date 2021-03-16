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

package org.datavec.image.recordreader.objdetect;

import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;

import org.junit.jupiter.api.Test;

import org.junit.jupiter.api.io.TempDir;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestVocLabelProvider {


    @Test
    public void testVocLabelProvider(@TempDir Path testDir) throws Exception {

        File f = testDir.toFile();
        new ClassPathResource("datavec-data-image/voc/2007/").copyDirectory(f);

        String path = f.getAbsolutePath();  //new ClassPathResource("voc/2007/JPEGImages/000005.jpg").getFile().getParentFile().getParent();

        ImageObjectLabelProvider lp = new VocLabelProvider(path);

        String img5 = new File(f, "JPEGImages/000005.jpg").getPath();

        List<ImageObject> l5 = lp.getImageObjectsForPath(img5);
        assertEquals(5, l5.size());

        List<ImageObject> exp5 = Arrays.asList(
                new ImageObject(263, 211, 324, 339, "chair"),
                new ImageObject(165, 264, 253, 372, "chair"),
                new ImageObject(5, 244, 67, 374, "chair"),
                new ImageObject(241, 194, 295, 299, "chair"),
                new ImageObject(277, 186, 312, 220, "chair"));

        assertEquals(exp5, l5);


        String img7 = new File(f, "JPEGImages/000007.jpg").getPath();
        List<ImageObject> exp7 = Collections.singletonList(new ImageObject(141, 50, 500, 330, "car"));

        assertEquals(exp7, lp.getImageObjectsForPath(img7));
    }

}
