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

package org.deeplearning4j.datasets.mnist.draw;

import java.io.FileInputStream;
import java.io.ObjectInputStream;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;



public class LoadAndDraw {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		MnistDataSetIterator iter = new MnistDataSetIterator(60,60000);
		@SuppressWarnings("unchecked")
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(args[0]));
		
		BasePretrainNetwork network = (BasePretrainNetwork) ois.readObject();
		
		
		DataSet test = null;
		while(iter.hasNext()) {
			test = iter.next();
			INDArray reconstructed = network.activate(test.getFeatureMatrix());
			for(int i = 0; i < test.numExamples(); i++) {
				INDArray draw1 = test.get(i).getFeatureMatrix().mul(255);
				INDArray reconstructed2 = reconstructed.getRow(i);
				INDArray draw2 = Nd4j.getDistributions().createBinomial(1,reconstructed2).sample(reconstructed2.shape()).mul(255);

				DrawReconstruction d = new DrawReconstruction(draw1);
				d.title = "REAL";
				d.draw();
				DrawReconstruction d2 = new DrawReconstruction(draw2,100,100);
				d2.title = "TEST";
				d2.draw();
				Thread.sleep(10000);
				d.frame.dispose();
				d2.frame.dispose();
			}
		}
		
		
	}

}
