/*
 *  * Copyright 2017 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.image.recordreader.objdetect;

import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.junit.Test;
import org.nd4j.linalg.io.ClassPathResource;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestVocLabelProvider {

    @Test
    public void testVocLabelProvider() throws Exception {

        String path = new ClassPathResource("voc/2007/JPEGImages/000005.jpg").getFile().getParentFile().getParent();

        ImageObjectLabelProvider lp = new VocLabelProvider(path);

        String img5 = new ClassPathResource("voc/2007/JPEGImages/000005.jpg").getFile().getPath();

        List<ImageObject> l5 = lp.getImageObjectsForPath(img5);
        assertEquals(5, l5.size());

        List<ImageObject> exp5 = Arrays.asList(
                new ImageObject(263, 211, 324, 339, "chair"),
                new ImageObject(165, 264, 253, 372, "chair"),
                new ImageObject(5, 244, 67, 374, "chair"),
                new ImageObject(241, 194, 295, 299, "chair"),
                new ImageObject(277, 186, 312, 220, "chair"));

        assertEquals(exp5, l5);


        String img7 = new ClassPathResource("voc/2007/JPEGImages/000007.jpg").getFile().getPath();
        List<ImageObject> exp7 = Collections.singletonList(new ImageObject(141, 50, 500, 330, "car"));

        assertEquals(exp7, lp.getImageObjectsForPath(img7));
    }

}
