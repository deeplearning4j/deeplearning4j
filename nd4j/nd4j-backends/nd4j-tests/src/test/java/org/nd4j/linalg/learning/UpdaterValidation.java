/* ******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.nd4j.linalg.learning;

import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.learning.config.*;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class UpdaterValidation extends BaseNd4jTest {

    public UpdaterValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testAdaDeltaUpdater(){
        double rho = 0.95;
        double epsilon = 1e-6;

        INDArray msg = Nd4j.zeros(DataType.DOUBLE, 1, 5);
        INDArray msdx = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("msg", msg.dup());
        state.put("msdx", msdx.dup());
        AdaDeltaUpdater u = (AdaDeltaUpdater) new AdaDelta(rho,epsilon).instantiate(state, true);

        assertEquals(msg, state.get("msg"));
        assertEquals(msdx, state.get("msdx"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyAdaDeltaUpdater(g1, msg, msdx, rho, epsilon);

            u.applyUpdater(g2, i, 0);

            assertEquals(msg, state.get("msg"));
            assertEquals(msdx, state.get("msdx"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testAdaGradUpdater(){
        double lr = 0.1;
        double epsilon = 1e-6;

        INDArray s = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("grad", s.dup());
        AdaGradUpdater u = (AdaGradUpdater) new AdaGrad(lr, epsilon).instantiate(state, true);

        assertEquals(s, state.get("grad"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyAdaGradUpdater(g1, s, lr, epsilon);

            u.applyUpdater(g2, i, 0);

            assertEquals(s, state.get("grad"));
            assertEquals(g1, g2);
        }
    }


    @Test
    public void testAdamUpdater(){

        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;

        INDArray m = Nd4j.zeros(DataType.DOUBLE, 1, 5);
        INDArray v = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("M", m.dup());
        state.put("V", v.dup());
        AdamUpdater u = (AdamUpdater) new Adam(lr, beta1, beta2, eps).instantiate(state, true);

        assertEquals(m, state.get("M"));
        assertEquals(v, state.get("V"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyAdamUpdater(g1, m, v, lr, beta1, beta2, eps, i);

            u.applyUpdater(g2, i, 0);

            assertEquals(m, state.get("M"));
            assertEquals(v, state.get("V"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testAdaMaxUpdater(){
        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;

        INDArray m = Nd4j.zeros(DataType.DOUBLE, 1, 5);
        INDArray v = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("M", m.dup());
        state.put("V", v.dup());
        AdaMaxUpdater u = (AdaMaxUpdater) new AdaMax(lr, beta1, beta2, eps).instantiate(state, true);

        assertEquals(m, state.get("M"));
        assertEquals(v, state.get("V"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyAdaMaxUpdater(g1, m, v, lr, beta1, beta2, eps, i);

            u.applyUpdater(g2, i, 0);

            assertEquals(m, state.get("M"));
            assertEquals(v, state.get("V"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testAmsGradUpdater(){
        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;

        INDArray m = Nd4j.zeros(DataType.DOUBLE, 1, 5);
        INDArray v = Nd4j.zeros(DataType.DOUBLE, 1, 5);
        INDArray vH = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("M", m.dup());
        state.put("V", v.dup());
        state.put("V_HAT", vH.dup());
        AMSGradUpdater u = (AMSGradUpdater) new AMSGrad(lr, beta1, beta2, eps).instantiate(state, true);

        assertEquals(m, state.get("M"));
        assertEquals(v, state.get("V"));
        assertEquals(vH, state.get("V_HAT"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyAmsGradUpdater(g1, m, v, vH, lr, beta1, beta2, eps, i);

            u.applyUpdater(g2, i, 0);

            assertEquals(m, state.get("M"));
            assertEquals(v, state.get("V"));
            assertEquals(vH, state.get("V_HAT"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testNadamUpdater(){

        double lr = 1e-3;
        double beta1 = 0.9;
        double beta2 = 0.999;
        double eps = 1e-8;

        INDArray m = Nd4j.zeros(DataType.DOUBLE, 1, 5);
        INDArray v = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("M", m.dup());
        state.put("V", v.dup());
        NadamUpdater u = (NadamUpdater) new Nadam(lr, beta1, beta2, eps).instantiate(state, true);

        assertEquals(m, state.get("M"));
        assertEquals(v, state.get("V"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyNadamUpdater(g1, m, v, lr, beta1, beta2, eps, i);

            u.applyUpdater(g2, i, 0);

            assertEquals(m, state.get("M"));
            assertEquals(v, state.get("V"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testNesterovUpdater(){

        double lr = 0.1;
        double momentum = 0.9;

        INDArray v = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("V", v.dup());
        NesterovsUpdater u = (NesterovsUpdater) new Nesterovs(lr, momentum).instantiate(state, true);

        assertEquals(v, state.get("V"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyNesterovsUpdater(g1, v, lr, momentum);

            u.applyUpdater(g2, i, 0);

            assertEquals(v, state.get("V"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testRmsPropUpdater(){

        double lr = 0.1;
        double decay = 0.95;
        double eps = 1e-8;

        INDArray g = Nd4j.zeros(DataType.DOUBLE, 1, 5);

        Map<String,INDArray> state = new HashMap<>();
        state.put("G", g.dup());
        RmsPropUpdater u = (RmsPropUpdater) new RmsProp(lr, decay, eps).instantiate(state, true);

        assertEquals(g, state.get("G"));

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applyRmsProp(g1, g, lr, decay, eps);

            u.applyUpdater(g2, i, 0);

            assertEquals(g, state.get("G"));
            assertEquals(g1, g2);
        }
    }

    @Test
    public void testSgdUpdater(){
        double lr = 0.1;

        SgdUpdater u = (SgdUpdater) new Sgd(lr).instantiate((Map<String,INDArray>)null, true);

        for( int i=0; i<3; i++ ) {
            INDArray g1 = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1,5);
            INDArray g2 = g1.dup();

            UpdaterJavaCode.applySgd(g1, lr);

            u.applyUpdater(g2, i, 0);
            assertEquals(g1, g2);
        }
    }


    /*
    @Test
    public void createUpdaterTestCases(){
        Nd4j.create(1);
        Nd4j.getRandom().setSeed(12345);

        int size = 5;

        for(boolean random : new boolean[]{false, true}) {
            System.out.println("/////////////////////////////// " + (random ? "RANDOM TEST CASES" : "LINSPACE TEST CASES") + " ///////////////////////////////" );

            for (IUpdater u : new IUpdater[]{new AdaDelta(), new Adam(), new AdaMax(), new AMSGrad(), new Nadam(), new Nesterovs(), new RmsProp(), new Sgd()}) {

                System.out.println(" ===== " + u + " =====");

                long ss = u.stateSize(size);
                INDArray state = ss > 0 ? Nd4j.create(DataType.DOUBLE, 1, ss) : null;
                GradientUpdater gu = u.instantiate(state, true);

                System.out.println("Initial state:");
                Map<String, INDArray> m = gu.getState();
                for (String s : m.keySet()) {
                    System.out.println("state: " + s + " - " + m.get(s).toStringFull());
                }

                for (int i = 0; i < 3; i++) {
                    System.out.println("Iteration: " + i);
                    INDArray in;
                    if(random){
                        in = Nd4j.rand(DataType.DOUBLE, 1, 5);
                    } else {
                        in = Nd4j.linspace(DataType.DOUBLE, 1, 5, 1).reshape(1, 5);
                    }

                    System.out.println("grad: " + in.toStringFull());
                    gu.applyUpdater(in, 0, 0);
                    System.out.println("update: " + in.toStringFull());

                    m = gu.getState();
                    for (String s : m.keySet()) {
                        System.out.println("state: " + s + " - " + m.get(s).toStringFull());
                    }
                }
            }
        }
    }
    */
}
