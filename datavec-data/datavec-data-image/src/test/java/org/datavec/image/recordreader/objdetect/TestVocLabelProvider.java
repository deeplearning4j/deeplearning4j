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

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.junit.Test;

public class TestVocLabelProvider {

    @Test
    public void testVoc() throws Exception {

        String path = new ClassPathResource("voc/2007/JPEGImages/000005.jpg").getFile().getParentFile().getParent();

        ImageObjectLabelProvider lp = new VocLabelProvider(path);

        String img5 = new ClassPathResource("voc/2007/JPEGImages/000005.jpg").getFile().getPath();

        lp.getImageObjectsForPath(img5);

    }

}
