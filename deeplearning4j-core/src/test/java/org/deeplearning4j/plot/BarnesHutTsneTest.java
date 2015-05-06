/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.deeplearning4j.plot;

import org.apache.commons.io.IOUtils;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.springframework.core.io.ClassPathResource;


import java.io.File;
import java.util.List;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class BarnesHutTsneTest {

    @Test
    public void testTsne() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        Nd4j.factory().setDType(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(250)
                .theta(0.5).learningRate(500).useAdaGrad(false)
                .build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getFile();
        INDArray data = Nd4j.readTxt(f.getAbsolutePath(),"   ").get(NDArrayIndex.interval(0,100),NDArrayIndex.interval(0,784));



        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0,100);
        b.fit(data);
    }


}
