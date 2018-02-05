/*-
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
import org.deeplearning4j.BaseDL4JTest;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.util.List;

import static org.junit.Assert.assertEquals;

// import org.nd4j.jita.conf.CudaEnvironment;

/**
 * Created by agibsonccc on 10/1/14.
 */
public class BarnesHutTsneTest extends BaseDL4JTest {
    @Before
    public void setUp() {
        //   CudaEnvironment.getInstance().getConfiguration().enableDebug(true).setVerbose(false);

    }



    @Test
    public void testTsne() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).setMaxIter(10).theta(0.5).learningRate(500)
                        .useAdaGrad(false).build();

        ClassPathResource resource = new ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ").get(NDArrayIndex.interval(0, 100),
                        NDArrayIndex.interval(0, 784));



        ClassPathResource labels = new ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0, 100);
        b.fit(data);
    }

    @Test
    public void testBuilderFields() throws Exception {
        final double theta = 0;
        final boolean invert = false;
        final String similarityFunctions = "euclidean";
        final int maxIter = 1;
        final double realMin = 1.0;
        final double initialMomentum = 2.0;
        final double finalMomentum = 3.0;
        final double momentum = 4.0;
        final int switchMomentumIteration = 1;
        final boolean normalize = false;
        final int stopLyingIteration = 100;
        final double tolerance = 1e-1;
        final double learningRate = 100;
        final boolean useAdaGrad = false;
        final double perplexity = 1.0;
        final double minGain = 1.0;

        BarnesHutTsne b = new BarnesHutTsne.Builder().theta(theta).invertDistanceMetric(invert)
                        .similarityFunction(similarityFunctions).setMaxIter(maxIter).setRealMin(realMin)
                        .setInitialMomentum(initialMomentum).setFinalMomentum(finalMomentum).setMomentum(momentum)
                        .setSwitchMomentumIteration(switchMomentumIteration).normalize(normalize)
                        .stopLyingIteration(stopLyingIteration).tolerance(tolerance).learningRate(learningRate)
                        .perplexity(perplexity).minGain(minGain).build();

        final double DELTA = 1e-15;

        assertEquals(theta, b.getTheta(), DELTA);
        assertEquals("invert", invert, b.isInvert());
        assertEquals("similarityFunctions", similarityFunctions, b.getSimiarlityFunction());
        assertEquals("maxIter", maxIter, b.maxIter);
        assertEquals(realMin, b.realMin, DELTA);
        assertEquals(initialMomentum, b.initialMomentum, DELTA);
        assertEquals(finalMomentum, b.finalMomentum, DELTA);
        assertEquals(momentum, b.momentum, DELTA);
        assertEquals("switchMomentumnIteration", switchMomentumIteration, b.switchMomentumIteration);
        assertEquals("normalize", normalize, b.normalize);
        assertEquals("stopLyingInMemoryLookupTable.javaIteration", stopLyingIteration, b.stopLyingIteration);
        assertEquals(tolerance, b.tolerance, DELTA);
        assertEquals(learningRate, b.learningRate, DELTA);
        assertEquals("useAdaGrad", useAdaGrad, b.useAdaGrad);
        assertEquals(perplexity, b.getPerplexity(), DELTA);
        assertEquals(minGain, b.minGain, DELTA);
    }
}
