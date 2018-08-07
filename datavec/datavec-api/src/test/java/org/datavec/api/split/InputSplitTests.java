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

package org.datavec.api.split;

import com.google.common.io.Files;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PatternPathLabelGenerator;
import org.junit.Test;

import java.io.*;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Random;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;

/**
 *
 * @author saudet
 */
public class InputSplitTests {

    @Test
    public void testSample() throws URISyntaxException {
        BaseInputSplit split = new BaseInputSplit() {
            {
                String[] paths = {"label0/group1_img.tif", "label1/group1_img.jpg", "label2/group1_img.png",
                                "label3/group1_img.jpeg", "label4/group1_img.bmp", "label5/group1_img.JPEG",
                                "label0/group2_img.JPG", "label1/group2_img.TIF", "label2/group2_img.PNG",
                                "label3/group2_img.jpg", "label4/group2_img.jpg", "label5/group2_img.wtf"};

                uriStrings = new ArrayList<>(paths.length);
                for (String s : paths) {
                    uriStrings.add("file:///" + s);
                }
            }

            @Override
            public void updateSplitLocations(boolean reset) {

            }

            @Override
            public boolean needsBootstrapForWrite() {
                return false;
            }

            @Override
            public void bootStrapForWrite() {

            }

            @Override
            public OutputStream openOutputStreamFor(String location) throws Exception {
                return null;
            }

            @Override
            public InputStream openInputStreamFor(String location) throws Exception {
                return null;
            }

            @Override
            public void reset() {
                //No op
            }

            @Override
            public boolean resetSupported() {
                return true;
            }

        };

        Random random = new Random(42);
        String[] extensions = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};
        ParentPathLabelGenerator parentPathLabelGenerator = new ParentPathLabelGenerator();
        PatternPathLabelGenerator patternPathLabelGenerator = new PatternPathLabelGenerator("_", 0);
        RandomPathFilter randomPathFilter = new RandomPathFilter(random, extensions);
        RandomPathFilter randomPathFilter2 = new RandomPathFilter(random, extensions, 7);
        BalancedPathFilter balancedPathFilter =
                        new BalancedPathFilter(random, extensions, parentPathLabelGenerator, 0, 4, 0, 1);
        BalancedPathFilter balancedPathFilter2 =
                        new BalancedPathFilter(random, extensions, patternPathLabelGenerator, 0, 4, 0, 1);

        InputSplit[] samples = split.sample(randomPathFilter);
        assertEquals(1, samples.length);
        assertEquals(11, samples[0].length());

        InputSplit[] samples2 = split.sample(randomPathFilter2);
        assertEquals(1, samples2.length);
        assertEquals(7, samples2[0].length());

        InputSplit[] samples3 = split.sample(balancedPathFilter, 80, 20);
        assertEquals(2, samples3.length);
        assertEquals(3, samples3[0].length());
        assertEquals(1, samples3[1].length());

        InputSplit[] samples4 = split.sample(balancedPathFilter2, 50, 50);
        assertEquals(2, samples4.length);
        assertEquals(1, samples4[0].length());
        assertEquals(1, samples4[1].length());
    }


    @Test
    public void testFileSplitBootstrap() {
        File tmpDir = Files.createTempDir();
        FileSplit boostrap = new FileSplit(tmpDir);
        assertTrue(boostrap.needsBootstrapForWrite());
        boostrap.bootStrapForWrite();
        assertTrue(tmpDir.listFiles() != null);
    }

}
