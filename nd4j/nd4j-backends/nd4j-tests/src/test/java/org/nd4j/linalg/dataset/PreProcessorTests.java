/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.dataset;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.LabelLastTimeStepPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;

import static org.junit.Assert.*;

public class PreProcessorTests extends BaseNd4jTest {

    public PreProcessorTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testLabelLastTimeStepPreProcessor(){

        INDArray f = Nd4j.rand(DataType.FLOAT, 3, 5, 8);
        INDArray l = Nd4j.rand(DataType.FLOAT, 3, 4, 8);

        //First test: no mask
        DataSet dsNoMask = new DataSet(f, l);

        DataSetPreProcessor preProc = new LabelLastTimeStepPreProcessor();
        preProc.preProcess(dsNoMask);

        assertSame(f, dsNoMask.getFeatures()); //Should be exact same object (not modified)

        INDArray l2d = dsNoMask.getLabels();
        INDArray l2dExp = l.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.point(7));
        assertEquals(l2dExp, l2d);


        //Second test: mask, but only 1 value at last time step


        INDArray lmSingle = Nd4j.createFromArray(new float[][]{
                {0,0,0,1,0,0,0,0},
                {0,0,0,1,0,0,1,0},
                {0,0,0,0,0,0,0,1}});

        INDArray fm = Nd4j.createFromArray(new float[][]{
                {1,1,1,1,0,0,0,0},
                {1,1,1,1,1,1,1,0},
                {1,1,1,1,1,1,1,1}});

        DataSet dsMask1 = new DataSet(f, l, fm, lmSingle);
        preProc.preProcess(dsMask1);

        INDArray expL = Nd4j.create(DataType.FLOAT, 3, 4);
        expL.putRow(0, l.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(3)));
        expL.putRow(1, l.get(NDArrayIndex.point(1), NDArrayIndex.all(), NDArrayIndex.point(6)));
        expL.putRow(2, l.get(NDArrayIndex.point(2), NDArrayIndex.all(), NDArrayIndex.point(7)));

        DataSet exp1 = new DataSet(f, expL, fm, null);
        assertEquals(exp1, dsMask1);

        //Third test: mask, but multiple values in label mask
        INDArray lmMultiple = Nd4j.createFromArray(new float[][]{
                {1,1,1,1,0,0,0,0},
                {1,1,1,1,1,1,1,0},
                {1,1,1,1,1,1,1,1}});

        DataSet dsMask2 = new DataSet(f, l, fm, lmMultiple);
        preProc.preProcess(dsMask2);
    }

    @Override
    public char ordering() {
        return 'c';
    }

}
