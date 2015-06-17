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

package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.List;

import static org.junit.Assert.assertEquals;

public class DataSetTest extends BaseNd4jTest {
	public DataSetTest() {
	}

	public DataSetTest(String name) {
		super(name);
	}

	public DataSetTest(String name, Nd4jBackend backend) {
		super(name, backend);
	}

	public DataSetTest(Nd4jBackend backend) {
		super(backend);
	}

	@Test
	public void testSetNewLabels() {
        Nd4j.MAX_ELEMENTS_PER_SLICE = Integer.MAX_VALUE;
        Nd4j.MAX_SLICES_TO_PRINT = Integer.MAX_VALUE;

		INDArray data = Nd4j.rand(10,10);
		INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{0, 0, 1, 1, 2, 2, 3, 3, 3, 3}, 4);
		DataSet d = new DataSet(data,labels);
		DataSet filtered = d.filterBy(new int[]{2,3});
		d.filterAndStrip(new int[]{2,3});
		assertEquals(getFailureMessage(),2,d.numOutcomes());
		assertEquals(getFailureMessage(),filtered.numExamples(),d.numExamples());
		assertEquals(getFailureMessage(),filtered.getFeatureMatrix(),d.getFeatureMatrix());
        assertEquals(getFailureMessage(),filtered.numExamples(), d.getLabels().rows());


	}

	@Test
	public void testAsListMmul() {
        System.out.println("/////////////");
        INDArray colVector = Nd4j.create(new float[]{1.0f,2.0f,3.0f,4.0f},new int[]{4,1});

        //First test: (works)
        //Get first Iris example
        DataSet x0 = new IrisDataSetIterator(1,1).next();
        INDArray example0 = x0.getFeatureMatrix();
        checkMMultDotProduct(example0,colVector);

        System.out.println("/////////////");

        //Second test: (fails)
        //Get first Iris example, but via asList()
        DataSet iris = new IrisDataSetIterator(150,150).next();
        List<DataSet> all = iris.asList();
        DataSet x1 = all.get(0);
        INDArray example1 = x1.getFeatureMatrix();
        System.out.println("example0 and example1 are equal: " + example0.equals(example1));
        checkMMultDotProduct(example1, colVector);
	}


    private  void checkMMultDotProduct(INDArray rowVector,INDArray colVector){
        assertTrue(rowVector.isRowVector());
        assertTrue(colVector.isColumnVector());
        System.out.println("rowVector: " + rowVector);
        System.out.println("colVector: " + colVector );
        float[] rowFloat = asFloat(rowVector);
        float[] colFloat = asFloat(colVector);

        INDArray product = rowVector.mmul(colVector);
        assertTrue(product.length() == 1);

        float expected = dotProduct(colFloat, rowFloat);
        float actual = product.getFloat(0);
        System.out.println("rowVector times colVector: expected = " + expected + ", actual = " + actual );
        assertEquals(getFailureMessage(),expected, actual, 0.01f);
    }


    public static float[] asFloat( INDArray arr ){
        int len = arr.length();
        float[] f = new float[len];
        for( int i=0; i < len; i++ )
            f[i] = arr.getFloat(i);
        return f;
    }

    public static float dotProduct( float[] x, float[] y ){
        float sum = 0.0f;
        for( int i = 0; i < x.length; i++ )
            sum += x[i] * y[i];
        return sum;
    }

    @Test
    public void testSplitTestAndTrain() throws Exception{
        INDArray labels = FeatureUtil.toOutcomeMatrix(new int[]{0,0,0,0,0,0,0,0},1);
        DataSet data = new DataSet(Nd4j.rand(8,1),labels);

        SplitTestAndTrain train = data.splitTestAndTrain(6, new DefaultRandom(1));
        assertEquals(train.getTrain().getLabels().length(),6);

        SplitTestAndTrain train2 = data.splitTestAndTrain(6, new DefaultRandom(1));
        assertEquals(getFailureMessage(),train.getTrain().getFeatureMatrix(), train2.getTrain().getFeatureMatrix());
    }


    @Override
    public char ordering() {
        return 'f';
    }
}
