/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.dimensionalityreduction;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.nd4j.linalg.dimensionalityreduction.RandomProjection.johnsonLindenStraussMinDim;
import static org.nd4j.linalg.dimensionalityreduction.RandomProjection.targetShape;

/**
 * Created by huitseeker on 7/28/17.
 */
@RunWith(Parameterized.class)
public class TestRandomProjection extends BaseNd4jTest {

    INDArray z1 = Nd4j.createUninitialized(new int[]{(int)1e6, 1000});

    @Rule
    public final ExpectedException exception = ExpectedException.none();


    public TestRandomProjection(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testJohnsonLindenStraussDim() {
        assertEquals(663, (int)johnsonLindenStraussMinDim((int) 1e6, 0.5).get(0));
        assertTrue(johnsonLindenStraussMinDim((int) 1e6, 0.5).equals(new ArrayList<Integer>(Arrays.asList(663))));

        ArrayList<Integer> res1 = new ArrayList<Integer>(Arrays.asList(663, 11841, 1112658));
        assertEquals(johnsonLindenStraussMinDim((int) 1e6, 0.5, 0.1, 0.01), res1);

        ArrayList<Integer> res2 = new ArrayList<>(Arrays.asList(7894,  9868, 11841));
        assertEquals(RandomProjection.johnsonLindenstraussMinDim(new int[]{(int) 1e4, (int) 1e5, (int) 1e6}, 0.1), res2);

    }

    @Test
    public void testTargetShape() {
        assertArrayEquals(targetShape(z1, 0.5), new long[]{1000, 663});
        assertArrayEquals(targetShape(Nd4j.createUninitialized(new int[]{(int)1e2, 225}), 0.5), new long[]{225, 221});
        // non-changing estimate
        assertArrayEquals(targetShape(z1, 700), new long[]{1000, 700});
    }

    @Test
    public void testTargetEpsilonChecks() {
        exception.expect(IllegalArgumentException.class);
        // wrong rel. error
        targetShape(z1, 0.0);
    }

    @Test
    public void testTargetShapeTooHigh() {
        exception.expect(ND4JIllegalStateException.class);
        // original dimension too small
        targetShape(Nd4j.createUninitialized(new int[]{(int)1e2, 1}), 0.5);
        // target dimension too high
        targetShape(z1, 1001);
        // suggested dimension too high
        targetShape(z1, 0.1);
        // original samples too small
        targetShape(Nd4j.createUninitialized(new int[]{1, 1000}), 0.5);
    }


    private void makeRandomSparseData(int[] shape, double density) {
        INDArray z1 = Nd4j.rand(shape);
        // because this is rand with mean = 0, stdev = 1, abslessThan ~= density
        BooleanIndexing.replaceWhere(z1, 0.0, Conditions.absLessThan(density));
    }


    private void testRandomProjectionDeterministicForSameShape(){
        INDArray z1 = Nd4j.randn(1000, 500);
        RandomProjection rp = new RandomProjection(50);
        INDArray res1 = Nd4j.zeros(10000, 442);
        rp.projecti(z1, res1);

        INDArray res2 = Nd4j.zeros(10000, 442);
        rp.projecti(z1, res2);

        assertEquals(res1, res2);
    }

    @Test
    public void testBasicEmbedding() {
        INDArray z1 = Nd4j.randn(10000, 500);
        RandomProjection rp = new RandomProjection(0.5);
        INDArray res = Nd4j.zeros(10000, 442);
        INDArray z2 = rp.projecti(z1, res);
        assertArrayEquals(new long[]{10000, 442}, z2.shape());
    }

    @Test
    public void testEmbedding(){
        INDArray z1 = Nd4j.randn(2000, 400);
        INDArray z2 = z1.dup();
        INDArray result = Transforms.allEuclideanDistances(z1, z2, 1);

        RandomProjection rp = new RandomProjection(0.5);
        INDArray zp = rp.project(z1);
        INDArray zp2 = zp.dup();
        INDArray projRes = Transforms.allEuclideanDistances(zp, zp2, 1);

        // check that the automatically tuned values for the density respect the
        // contract for eps: pairwise distances are preserved according to the
        // Johnson-Lindenstrauss lemma
        INDArray ratios = projRes.div(result);

        for (int i = 0; i < ratios.length(); i++){
            double val = ratios.getDouble(i);
            // this avoids the NaNs we get along the diagonal
            if (val == val) {
                assertTrue(ratios.getDouble(i) < 1.5);
            }
        }

    }


    @Override
    public char ordering() {
        return 'f';
    }

}
