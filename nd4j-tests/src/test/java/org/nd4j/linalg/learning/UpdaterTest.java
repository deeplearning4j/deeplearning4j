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

package org.nd4j.linalg.learning;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

public class UpdaterTest extends BaseNd4jTest {

	private static final Logger log = LoggerFactory.getLogger(UpdaterTest.class);

	public UpdaterTest(String name, Nd4jBackend backend) {
		super(name, backend);
	}

    public UpdaterTest() {
    }

    public UpdaterTest(Nd4jBackend backend) {
        super(backend);
    }

    public UpdaterTest(String name) {
        super(name);
    }

    @Test
	public void testAdaGrad1() {
		int rows = 1;
		int cols = 1;


		AdaGrad grad = new AdaGrad(rows,cols,1e-3);
		INDArray W = Nd4j.ones(rows, cols);
		assertEquals(1e-1,grad.getGradient(W,0).getDouble(0),1e-1);



	}
	@Test
	public void testNesterovs() {
		int rows = 10;
		int cols = 2;


		Nesterovs grad = new Nesterovs(0.5);
		INDArray W = Nd4j.zeros(rows, cols);
		Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
		for(int i = 0; i < W.rows(); i++)
			W.putRow(i, Nd4j.create(dist.sample(W.columns())));

		for(int i = 0; i < 5; i++) {
			String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W,i)).replaceAll(";","\n");
			System.out.println(learningRates);
			W.addi(Nd4j.randn(rows, cols));
		}

	}


	@Test
	public void testAdaGrad() {
		int rows = 10;
		int cols = 2;


		AdaGrad grad = new AdaGrad(rows,cols,0.1);
		INDArray W = Nd4j.zeros(rows, cols);
		Distribution dist = Nd4j.getDistributions().createNormal(1, 1);
		for(int i = 0; i < W.rows(); i++)
			W.putRow(i, Nd4j.create(dist.sample(W.columns())));

		for(int i = 0; i < 5; i++) {
			String learningRates = String.valueOf("\nAdagrad\n " + grad.getGradient(W,i)).replaceAll(";","\n");
			System.out.println(learningRates);
			W.addi(Nd4j.randn(rows, cols));
		}

	}

	@Test
	public void testAdaDelta() {
		int rows = 10;
		int cols = 2;


		AdaDelta grad = new AdaDelta();
		INDArray W = Nd4j.zeros(rows, cols);
		Distribution dist = Nd4j.getDistributions().createNormal(1e-3,1e-3);
		for(int i = 0; i < W.rows(); i++)
			W.putRow(i, Nd4j.create(dist.sample(W.columns())));

		for(int i = 0; i < 5; i++) {
			String learningRates = String.valueOf("\nAdaelta\n " + grad.getGradient(W,i)).replaceAll(";","\n");
			System.out.println(learningRates);
			W.addi(Nd4j.randn(rows, cols));
		}

	}

	@Test
	public void testAdam() {
		int rows = 10;
		int cols = 2;


		Adam grad = new Adam();
		INDArray W = Nd4j.zeros(rows, cols);
		Distribution dist = Nd4j.getDistributions().createNormal(1e-3,1e-3);
		for(int i = 0; i < W.rows(); i++)
			W.putRow(i, Nd4j.create(dist.sample(W.columns())));

		for(int i = 0; i < 5; i++) {
			String learningRates = String.valueOf("\nAdam\n " + grad.getGradient(W,i)).replaceAll(";","\n");
			System.out.println(learningRates);
			W.addi(Nd4j.randn(rows, cols));
		}

	}

	@Override
	public char ordering() {
		return 'f';
	}
}
