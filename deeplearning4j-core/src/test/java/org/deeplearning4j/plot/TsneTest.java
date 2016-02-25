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
import org.canova.api.util.ClassPathResource;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;


import java.io.File;
import java.util.List;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class TsneTest {

    @Test
    public void testTsne() throws Exception {
        Tsne calculation = new Tsne.Builder()
                .setMaxIter(10)
                .usePca(false)
                .setSwitchMomentumIteration(20)
                .normalize(false)
                .useAdaGrad(false)
                .learningRate(500)
                .perplexity(20)
                .minGain(1e-1f)
                .build();
        //ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = new File("/ext/mnist2500_X.txt");
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(),"   ");//.get(NDArrayIndex.interval(0,600));
        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0,3);

        calculation.calculate(data, 2, 20);


    }

    @Test
    @Ignore
    public void testExternalTsne() throws Exception {
        ProperTsne calculation = new ProperTsne.Builder()
                .setMaxIter(20)
                .usePca(false)
                .setSwitchMomentumIteration(20)
                .normalize(false)
                .useAdaGrad(false)
                .learningRate(500)
                .perplexity(20)
                .minGain(1e-1f)
                .perplexity(20)
                .build();
        //ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = new File("/ext/mnist2500_X.txt");
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(),"   "); //.get(NDArrayIndex.interval(0,600));
        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream());//.subList(0,3);

        //calculation.calculate(data, 2, 20);
        calculation.plot(data, 2, labelsList, "/ext/Temp/coords.txt");


    }

}
