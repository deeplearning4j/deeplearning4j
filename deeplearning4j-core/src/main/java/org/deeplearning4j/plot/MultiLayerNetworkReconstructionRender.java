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

import org.deeplearning4j.datasets.mnist.draw.DrawReconstruction;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Reconstruction renders for a multi layer network
 */
public class MultiLayerNetworkReconstructionRender {

    private DataSetIterator iter;
    private MultiLayerNetwork network;
    private int reconLayer = -1;

    public MultiLayerNetworkReconstructionRender(DataSetIterator iter, MultiLayerNetwork network,int reconLayer) {
        this.iter = iter;
        this.network = network;
        this.reconLayer = reconLayer;
    }
    public MultiLayerNetworkReconstructionRender(DataSetIterator iter, MultiLayerNetwork network) {
        this(iter,network,-1);
    }
    public void draw() throws InterruptedException {
        while(iter.hasNext()) {
            DataSet first = iter.next();
            INDArray reconstruct = null;
            if(reconLayer < 0)
                reconstruct = network.output(first.getFeatureMatrix());
             else
                  reconstruct = network.reconstruct(first.getFeatureMatrix(),reconLayer);
            for(int j = 0; j < first.numExamples(); j++) {

                INDArray draw1 = first.get(j).getFeatureMatrix().mul(255);
                INDArray reconstructed2 = reconstruct.getRow(j);
                INDArray draw2 = reconstructed2.mul(255 * 255);

                DrawReconstruction d = new DrawReconstruction(draw1);
                d.title = "REAL";
                d.draw();
                DrawReconstruction d2 = new DrawReconstruction(draw2);
                d2.title = "TEST";
                d2.draw();
                Thread.sleep(10000);
                d.frame.dispose();
                d2.frame.dispose();
            }


        }
    }


}
