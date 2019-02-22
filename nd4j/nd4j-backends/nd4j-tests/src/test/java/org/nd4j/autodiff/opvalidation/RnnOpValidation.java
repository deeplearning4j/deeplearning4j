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

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMBlockCellConfiguration;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

@Slf4j
public class RnnOpValidation extends BaseOpValidation {
    public RnnOpValidation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testRnnBlockCell(){
        Nd4j.getRandom().setSeed(12345);
        int mb = 2;
        int nIn = 3;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.constant(Nd4j.rand(DataType.FLOAT, mb, nIn));
        SDVariable cLast = sd.constant(Nd4j.rand(DataType.FLOAT, mb, nOut));
        SDVariable yLast = sd.constant(Nd4j.rand(DataType.FLOAT, mb, nOut));
        SDVariable W = sd.constant(Nd4j.rand(DataType.FLOAT, (nIn+nOut), 4*nOut));
        SDVariable Wci = sd.constant(Nd4j.rand(DataType.FLOAT, nOut));
        SDVariable Wcf = sd.constant(Nd4j.rand(DataType.FLOAT, nOut));
        SDVariable Wco = sd.constant(Nd4j.rand(DataType.FLOAT, nOut));
        SDVariable b = sd.constant(Nd4j.rand(DataType.FLOAT, 4*nOut));

        double fb = 1.0;
        LSTMBlockCellConfiguration conf = LSTMBlockCellConfiguration.builder()
                .xt(x)
                .cLast(cLast)
                .yLast(yLast)
                .W(W)
                .Wci(Wci)
                .Wcf(Wcf)
                .Wco(Wco)
                .b(b)
                .peepHole(true)
                .forgetBias(fb)
                .clippingCellValue(0.0)
                .build();

        List<SDVariable> v = sd.rnn().lstmBlock("lstm", conf);  //Output order: i, c, f, o, z, h, y
        List<String> toExec = new ArrayList<>();
        for(SDVariable sdv : v){
            toExec.add(sdv.getVarName());
        }

        //Test forward pass:
        Map<String,INDArray> m = sd.exec(null, toExec);

        //Weights and bias order: [i, f, z, o]

        //Block input (z) - post tanh:
        INDArray wz_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(2*nOut, 3*nOut));           //Input weights
        INDArray wz_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(2*nOut, 3*nOut));    //Recurrent weights
        INDArray bz = b.getArr().get(NDArrayIndex.interval(2*nOut, 3*nOut));

        INDArray zExp = x.getArr().mmul(wz_x).addiRowVector(bz);        //[mb,nIn]*[nIn, nOut] + [nOut]
        zExp.addi(yLast.getArr().mmul(wz_r));   //[mb,nOut]*[nOut,nOut]
        Transforms.tanh(zExp, false);

        INDArray zAct = m.get(toExec.get(4));
        assertEquals(zExp, zAct);

        //Input modulation gate (post sigmoid) - i: (note: peephole input - last time step)
        INDArray wi_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(0, nOut));           //Input weights
        INDArray wi_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(0, nOut));    //Recurrent weights
        INDArray bi = b.getArr().get(NDArrayIndex.interval(0, nOut));

        INDArray iExp = x.getArr().mmul(wi_x).addiRowVector(bi);        //[mb,nIn]*[nIn, nOut] + [nOut]
        iExp.addi(yLast.getArr().mmul(wi_r));   //[mb,nOut]*[nOut,nOut]
        iExp.addi(cLast.getArr().mulRowVector(Wci.getArr()));    //Peephole
        Transforms.sigmoid(iExp, false);
        assertEquals(iExp, m.get(toExec.get(0)));

        //Forget gate (post sigmoid): (note: peephole input - last time step)
        INDArray wf_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(1*nOut, 2*nOut));           //Input weights
        INDArray wf_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(1*nOut, 2*nOut));    //Recurrent weights
        INDArray bf = b.getArr().get(NDArrayIndex.interval(1*nOut, 2*nOut));

        INDArray fExp = x.getArr().mmul(wf_x).addiRowVector(bf);        //[mb,nIn]*[nIn, nOut] + [nOut]
        fExp.addi(yLast.getArr().mmul(wf_r));   //[mb,nOut]*[nOut,nOut]
        fExp.addi(cLast.getArr().mulRowVector(Wcf.getArr()));   //Peephole
        fExp.addi(fb);
        Transforms.sigmoid(fExp, false);
        assertEquals(fExp, m.get(toExec.get(2)));

        //Cell state (pre tanh): tanh(z) .* sigmoid(i) + sigmoid(f) .* cLast
        INDArray cExp = zExp.mul(iExp).add(fExp.mul(cLast.getArr()));
        INDArray cAct = m.get(toExec.get(1));
        assertEquals(cExp, cAct);

        //Output gate (post sigmoid): (note: peephole input: current time step)
        INDArray wo_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(3*nOut, 4*nOut));           //Input weights
        INDArray wo_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(3*nOut, 4*nOut));    //Recurrent weights
        INDArray bo = b.getArr().get(NDArrayIndex.interval(3*nOut, 4*nOut));

        INDArray oExp = x.getArr().mmul(wo_x).addiRowVector(bo);        //[mb,nIn]*[nIn, nOut] + [nOut]
        oExp.addi(yLast.getArr().mmul(wo_r));   //[mb,nOut]*[nOut,nOut]
        oExp.addi(cExp.mulRowVector(Wco.getArr())); //Peephole
        Transforms.sigmoid(oExp, false);
        assertEquals(oExp, m.get(toExec.get(3)));

        //Cell state, post tanh
        INDArray hExp = Transforms.tanh(cExp, true);
        assertEquals(hExp, m.get(toExec.get(5)));

        //Final output
        INDArray yExp = hExp.mul(oExp);
        assertEquals(yExp, m.get(toExec.get(6)));
    }
}