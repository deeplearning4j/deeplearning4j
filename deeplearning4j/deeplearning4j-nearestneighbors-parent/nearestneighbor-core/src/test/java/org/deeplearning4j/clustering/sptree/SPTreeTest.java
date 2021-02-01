/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.clustering.sptree;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.BaseDL4JTest;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.util.concurrent.AtomicDouble;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class SPTreeTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }

    @Before
    public void setUp() {
        DataTypeUtil.setDTypeForContext(DataType.DOUBLE);
    }

    @Test
    public void testStructure() {
        INDArray data = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        SpTree tree = new SpTree(data);
        /*try (MemoryWorkspace ws = tree.workspace().notifyScopeEntered())*/ {
            assertEquals(Nd4j.create(new double[]{2.5f, 3.5f, 4.5f}), tree.getCenterOfMass());
            assertEquals(2, tree.getCumSize());
            assertEquals(8, tree.getNumChildren());
            assertTrue(tree.isCorrect());
        }
    }

    @Test
    public void testComputeEdgeForces() {
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        double[] aData = new double[]{
                0.2999816948164936, 0.26252049735806526, 0.2673853427498767, 0.8604464129156685, 0.4802652829902563, 0.10959096539488711, 0.7950242948008909, 0.5917848948003486,
                0.2738285999345498, 0.9519684328285567, 0.9690024759209738, 0.8585615547624705, 0.8087760944312002, 0.5337951589543348, 0.5960876109129123, 0.7187130179825856,
                0.4629777327445964, 0.08665909175584818, 0.7748005397731237, 0.48020186965468536, 0.24927351841378798, 0.32272599988270445, 0.306414968984427, 0.6980212149215657,
                0.7977183964212472, 0.7673513094629704, 0.1679681724796478, 0.3107359484804584, 0.021701726051792103, 0.13797462786662518, 0.8618953518813538, 0.841333838365635,
                0.5284957375170422, 0.9703367685039823, 0.677388096913733, 0.2624474979832243, 0.43740966353106536, 0.15685545957858893, 0.11072929134449871, 0.06007395961283357,
                0.4093918718557811,  0.9563909195720572, 0.5994144944480242, 0.8278927844215804, 0.38586830957105667, 0.6201844716257464, 0.7603829079070265, 0.07875691596842949,
                0.08651136699915507, 0.7445210640026082, 0.6547649514127559, 0.3384719042666908, 0.05816723105860,0.6248951423054205, 0.7431868493349041};
        INDArray data = Nd4j.createFromArray(aData).reshape(11,5);
        INDArray rows = Nd4j.createFromArray(new int[]{
                         0,         9,        18,        27,        36,        45,        54,        63,        72,        81,        90,        99});
        INDArray cols = Nd4j.createFromArray(new int[]{
                4,         3,        10,         8,         6,         7,         1,         5,         9,         4,         9,         8,        10,         2,         0,         6,         7,         3,         6,         8,         3,         9,        10,         1,         4,         0,         5,        10,         0,         4,         6,         8,         9,         2,         5,         7,         0,        10,         3,         1,         8,         9,         6,         7,         2,         7,         9,         3,        10,         0,         4,         2,         8,         1,         2,         8,         3,        10,         0,         4,         9,         1,         5,         5,         9,         0,         3,        10,         4,         8,         1,         2,         6,         2,         0,         3,         4,         1,        10,         9,         7,        10,         1,         3,         7,         4,         5,         2,         8,         6,         3,         4,         0,         9,         6,         5,         8,         7,         1});
        INDArray vals = Nd4j.createFromArray(new double[]
                {    0.6806,    0.1978,    0.1349,    0.0403,    0.0087,    0.0369,    0.0081,    0.0172,    0.0014,    0.0046,    0.0081,    0.3375,    0.2274,    0.0556,    0.0098,    0.0175,    0.0027,    0.0077,    0.0014,    0.0023,    0.0175,    0.6569,    0.1762,    0.0254,    0.0200,    0.0118,    0.0074,    0.0046,    0.0124,    0.0012,    0.1978,    0.0014,    0.0254,    0.7198,    0.0712,    0.0850,    0.0389,    0.0555,    0.0418,    0.0286,    0.6806,    0.3375,    0.0074,    0.0712,    0.2290,    0.0224,    0.0189,    0.0080,    0.0187,    0.0097,    0.0172,    0.0124,    0.0418,    0.7799,    0.0521,    0.0395,    0.0097,    0.0030,    0.0023,  1.706e-5,    0.0087,    0.0027,    0.6569,    0.0850,    0.0080,    0.5562,    0.0173,    0.0015,  1.706e-5,    0.0369,    0.0077,    0.0286,    0.0187,    0.7799,    0.0711,    0.0200,    0.0084,    0.0012,    0.0403,    0.0556,    0.1762,    0.0389,    0.0224,    0.0030,    0.5562,    0.0084,    0.0060,    0.0028,    0.0014,    0.2274,    0.0200,    0.0555,    0.0189,    0.0521,    0.0015,    0.0711,    0.0028,    0.3911,    0.1349,    0.0098,    0.0118,    0.7198,    0.2290,    0.0395,    0.0173,    0.0200,    0.0060,    0.3911});
        SpTree tree = new SpTree(data);
        INDArray posF = Nd4j.create(11, 5);
        /*try (MemoryWorkspace ws = tree.workspace().notifyScopeEntered())*/ {
            tree.computeEdgeForces(rows, cols, vals, 11, posF);
        }
        INDArray expected = Nd4j.createFromArray(new double[]{     -0.08045664291717945, -0.1010737980370276, 0.01793326162563703, 0.16108447776416351, -0.20679423033936287, -0.15788549368713395, 0.02546624825966788, 0.062309466206907055, -0.165806093080134, 0.15266225270841186, 0.17508365896345726, 0.09588570563583201, 0.34124767300538084, 0.14606666020839956, -0.06786563815470595, -0.09326646571247202, -0.19896040730569928, -0.3618837364446506, 0.13946315445146712, -0.04570186310149667, -0.2473462951783839, -0.41362278505023914, -0.1094083777758208, 0.10705807646770374, 0.24462088260113946, 0.21722270026621748, -0.21799892431326567, -0.08205544003080587, -0.11170161709042685, -0.2674768703060442, 0.03617747284043274, 0.16430316252598698, 0.04552845070022399, 0.2593696744801452, 0.1439989190892037, -0.059339471967457376, 0.05460893792863096, -0.0595168036583193, -0.2527693197519917, -0.15850951859835274, -0.2945536856938165, 0.15434659331638875, -0.022910846947667776, 0.23598009757792854, -0.11149279745674007, 0.09670616593772939, 0.11125703954547914, -0.08519984596392606, -0.12779827002328714, 0.23025192887225998, 0.13741473964038722, -0.06193553503816597, -0.08349781586292176, 0.1622156410642145, 0.155975447743472}).reshape(11,5);
        for (int i = 0; i < 11; ++i)
            assertArrayEquals(expected.getRow(i).toDoubleVector(), posF.getRow(i).toDoubleVector(), 1e-2);

        AtomicDouble sumQ = new AtomicDouble(0.0);
        /*try (MemoryWorkspace ws = tree.workspace().notifyScopeEntered())*/ {
            tree.computeNonEdgeForces(0, 0.5, Nd4j.zeros(5), sumQ);
        }
        assertEquals(8.65, sumQ.get(), 1e-2);
    }

    @Test
    //@Ignore
    public void testLargeTree() {
        int num = isIntegrationTests() ? 100000 : 1000;
        StopWatch watch = new StopWatch();
        watch.start();
        INDArray arr = Nd4j.linspace(1, num, num, Nd4j.dataType()).reshape(num, 1);
        SpTree tree = new SpTree(arr);
        watch.stop();
        System.out.println("Tree of size " + num + " created in " + watch);
    }

}
