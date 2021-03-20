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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.adapter.SingletonDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImageMultiPreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;


@Tag(TagNames.NDARRAY_ETL)
@NativeTag
public class ImagePreProcessortTest extends BaseNd4jTestWithBackends {

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void simpleImageTest(Nd4jBackend backend) {
        INDArray rChannels = Nd4j.zeros(DataType.FLOAT, 10, 10).addi(128);
        INDArray gChannels = Nd4j.zeros(DataType.FLOAT, 10, 10).addi(64);
        INDArray bChannels = Nd4j.zeros(DataType.FLOAT, 10, 10).addi(255);
        INDArray image = Nd4j.vstack(rChannels, gChannels, bChannels).reshape(1, 3, 10, 10);
        INDArray orig = image.dup();

        //System.out.println(Arrays.toString(image.shape()));
        DataSet ds = new DataSet(image, Nd4j.ones(1, 1));
        ImagePreProcessingScaler myScaler = new ImagePreProcessingScaler();
        //So this should scale to 0.5,0.25 and 1;
        INDArray expected = image.mul(0);
        expected.slice(0, 1).addi(0.5);
        expected.slice(1, 1).addi(0.25);
        expected.slice(2, 1).addi(1.0);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(expected)).maxNumber().doubleValue() <= 0.01);

        //Now giving it 16 bits instead of the default
        //System.out.println(Arrays.toString(image.shape()));
        ds = new DataSet(image, Nd4j.ones(1, 1));
        myScaler = new ImagePreProcessingScaler(0, 1, 16);
        //So this should scale to 0.5,0.25 and 1;
        expected = image.mul(0);
        expected.slice(0, 1).addi(0.5 / 256);
        expected.slice(1, 1).addi(0.25 / 256);
        expected.slice(2, 1).addi(1.0 / 256);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(expected)).maxNumber().doubleValue() <= 0.01);

        //So this should not change the value
        INDArray before = ds.getFeatures().dup();
        myScaler = new ImagePreProcessingScaler(0, 1, 1);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(before)).maxNumber().doubleValue() <= 0.0001);

        //Scaling back up should give the same results
        myScaler = new ImagePreProcessingScaler(0, (256 * 256 * 256 - 1), 1);
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures().sub(image)).maxNumber().doubleValue() <= 1);

        //Revert:
        before = orig.dup();
        myScaler = new ImagePreProcessingScaler(0, 1, 1);
        myScaler.transform(before);
        myScaler.revertFeatures(before);
        assertEquals(orig, before);


        //Test labels (segmentation case)
        before = orig.dup();
        myScaler = new ImagePreProcessingScaler(0, 1);
        myScaler.transformLabel(before);
        expected = orig.div(255);
        assertEquals(expected, before);
        myScaler.revertLabels(before);
        assertEquals(orig, before);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void simpleImageTestMulti(Nd4jBackend backend) {
        INDArray rChannels = Nd4j.zeros(10, 10).addi(128);
        INDArray gChannels = Nd4j.zeros(10, 10).addi(64);
        INDArray bChannels = Nd4j.zeros(10, 10).addi(255);
        INDArray image = Nd4j.vstack(rChannels, gChannels, bChannels).reshape(3, 10, 10);
        INDArray orig = image.dup();

        //System.out.println(Arrays.toString(image.shape()));
        MultiDataSet ds = new MultiDataSet(new INDArray[]{Nd4j.valueArrayOf(10, 100.0), image.reshape(1, 3, 10, 10)},
                new INDArray[]{Nd4j.ones(1, 1)});
        ImageMultiPreProcessingScaler myScaler = new ImageMultiPreProcessingScaler(1);
        //So this should scale to 0.5,0.25 and 1;
        INDArray expected = image.mul(0);
        expected.slice(0, 0).addi(0.5);
        expected.slice(1, 0).addi(0.25);
        expected.slice(2, 0).addi(1.0);
        myScaler.transform(ds);
        assertEquals(Nd4j.valueArrayOf(10, 100.0), ds.getFeatures(0));
        assertTrue(Transforms.abs(ds.getFeatures(1).sub(expected)).maxNumber().doubleValue() <= 0.01);

        //Now giving it 16 bits instead of the default
        //System.out.println(Arrays.toString(image.shape()));
        ds = new MultiDataSet(new INDArray[]{Nd4j.valueArrayOf(10, 100.0), image.reshape(1, 3, 10, 10)},
                new INDArray[]{Nd4j.ones(1, 1)});
        myScaler = new ImageMultiPreProcessingScaler(0.0, 1.0, 16, new int[]{1});
        //So this should scale to 0.5,0.25 and 1;
        expected = image.mul(0);
        expected.slice(0, 0).addi(0.5 / 256);
        expected.slice(1, 0).addi(0.25 / 256);
        expected.slice(2, 0).addi(1.0 / 256);
        myScaler.transform(ds);
        assertEquals(Nd4j.valueArrayOf(10, 100.0), ds.getFeatures(0));
        assertTrue(Transforms.abs(ds.getFeatures(1).sub(expected)).maxNumber().doubleValue() <= 0.01);

        //So this should not change the value
        INDArray before = ds.getFeatures(1).dup();
        myScaler = new ImageMultiPreProcessingScaler(0.0, 1.0, new int[]{1});
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures(1).sub(before)).maxNumber().doubleValue() <= 0.0001);

        //Scaling back up should give the same results
        myScaler = new ImageMultiPreProcessingScaler(0.0, (256.0 * 256 * 256 - 1), new int[]{1});
        myScaler.transform(ds);
        assertTrue(Transforms.abs(ds.getFeatures(1).sub(image)).maxNumber().doubleValue() <= 1);

        //Revert:
        before = orig.dup();
        myScaler = new ImageMultiPreProcessingScaler(0.0, 1.0, 1, new int[]{1});
        MultiDataSet beforeDS = new MultiDataSet(new INDArray[]{null, before}, new INDArray[]{null});
        myScaler.transform(beforeDS);
        myScaler.revertFeatures(beforeDS.getFeatures());
        assertEquals(orig, before);
    }


    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testSegmentation(Nd4jBackend backend){

        INDArray f = Nd4j.math().floor(Nd4j.rand(DataType.FLOAT, 3, 3, 16, 16).muli(255));
        INDArray l = Nd4j.math().floor(Nd4j.rand(DataType.FLOAT, 3, 10, 8, 8).muli(255));

        ImagePreProcessingScaler s = new ImagePreProcessingScaler();
        s.fitLabel(true);

        s.fit(new DataSet(f,l));

        INDArray expF = f.div(255);
        INDArray expL = l.div(255);

        DataSet d = new DataSet(f.dup(), l.dup());
        s.transform(d);
        assertEquals(expF, d.getFeatures());
        assertEquals(expL, d.getLabels());


        s.fit(new SingletonDataSetIterator(new DataSet(f, l)));

        INDArray f2 = f.dup();
        INDArray l2 = l.dup();
        s.transform(f2);
        s.transformLabel(l2);
        assertEquals(expF, f2);
        assertEquals(expL, l2);

        s.revertFeatures(f2);
        s.revertLabels(l2);
        assertEquals(f, f2);
        assertEquals(l, l2);
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
