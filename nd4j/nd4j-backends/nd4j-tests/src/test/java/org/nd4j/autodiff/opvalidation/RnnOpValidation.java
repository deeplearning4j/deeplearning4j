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

package org.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMCellOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.GRUWeights;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMWeights;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
        LSTMConfiguration conf = LSTMConfiguration.builder()
                .peepHole(true)
                .forgetBias(fb)
                .clippingCellValue(0.0)
                .build();

        LSTMWeights weights = LSTMWeights.builder().weights(W).bias(b)
                .inputPeepholeWeights(Wci).forgetPeepholeWeights(Wcf).outputPeepholeWeights(Wco).build();

        LSTMCellOutputs v = new LSTMCellOutputs(sd.rnn().lstmCell(x, cLast, yLast, weights, conf));  //Output order: i, c, f, o, z, h, y
        List<String> toExec = new ArrayList<>();
        for(SDVariable sdv : v.getAllOutputs()){
            toExec.add(sdv.name());
        }

        //Test forward pass:
        Map<String,INDArray> m = sd.output(null, toExec);

        //Weights and bias order: [i, f, z, o]

        //Block input (z) - post tanh:
        INDArray wz_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(nOut, 2*nOut));           //Input weights
        INDArray wz_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(nOut, 2*nOut));    //Recurrent weights
        INDArray bz = b.getArr().get(NDArrayIndex.interval(nOut, 2*nOut));

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
        INDArray wf_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(2*nOut, 3*nOut));           //Input weights
        INDArray wf_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(2*nOut, 3*nOut));    //Recurrent weights
        INDArray bf = b.getArr().get(NDArrayIndex.interval(2*nOut, 3*nOut));

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


    @Test
    public void testRnnBlockCellManualTFCompare() {
        //Test case: "rnn/lstmblockcell/static_batch1_n3-2_tsLength1_noPH_noClip_fBias1_noIS"

        SameDiff sd = SameDiff.create();
        INDArray zero2d = Nd4j.createFromArray(new float[][]{{0,0}});
        INDArray zero1d = Nd4j.createFromArray(new float[]{0,0});
        SDVariable x = sd.constant(Nd4j.createFromArray(new float[][]{{0.7787856f,0.80119777f,0.72437465f}}));
        SDVariable cLast = sd.constant(zero2d);
        SDVariable yLast = sd.constant(zero2d);
        //Weights shape: [(nIn+nOut), 4*nOut]
        SDVariable W = sd.constant(Nd4j.createFromArray(-0.61977,-0.5708851,-0.38089648,-0.07994056,-0.31706482,0.21500933,-0.35454142,-0.3239095,-0.3177906,
                0.39918554,-0.3115911,0.540841,0.38552666,0.34270835,-0.63456273,-0.13917702,-0.2985368,0.343238,
                -0.3178353,0.017154932,-0.060259163,0.28841054,-0.6257687,0.65097713,0.24375653,-0.22315514,0.2033832,
                0.24894875,-0.2062299,-0.2242794,-0.3809483,-0.023048997,-0.036284804,-0.46398938,-0.33979666,0.67012596,
                -0.42168984,0.34208286,-0.0456419,0.39803517).castTo(DataType.FLOAT).reshape(5,8));
        SDVariable Wci = sd.constant(zero1d);
        SDVariable Wcf = sd.constant(zero1d);
        SDVariable Wco = sd.constant(zero1d);
        SDVariable b = sd.constant(Nd4j.zeros(DataType.FLOAT, 8));

        double fb = 1.0;
        LSTMConfiguration conf = LSTMConfiguration.builder()
                .peepHole(false)
                .forgetBias(fb)
                .clippingCellValue(0.0)
                .build();

        LSTMWeights weights = LSTMWeights.builder().weights(W).bias(b)
                .inputPeepholeWeights(Wci).forgetPeepholeWeights(Wcf).outputPeepholeWeights(Wco).build();

        LSTMCellOutputs v = new LSTMCellOutputs(sd.rnn().lstmCell(x, cLast, yLast, weights, conf));  //Output order: i, c, f, o, z, h, y
        List<String> toExec = new ArrayList<>();
        for(SDVariable sdv : v.getAllOutputs()){
            toExec.add(sdv.name());
        }

        //Test forward pass:
        Map<String,INDArray> m = sd.output(null, toExec);

        INDArray out0 = Nd4j.create(new float[]{0.27817473f, 0.53092605f}, new int[]{1,2});     //Input mod gate
        INDArray out1 = Nd4j.create(new float[]{-0.18100877f, 0.19417824f}, new int[]{1,2});    //CS (pre tanh)
        INDArray out2 = Nd4j.create(new float[]{0.73464274f, 0.83901811f}, new int[]{1,2});     //Forget gate
        INDArray out3 = Nd4j.create(new float[]{0.22481689f, 0.52692068f}, new int[]{1,2});     //Output gate

        INDArray out4 = Nd4j.create(new float[]{-0.65070170f, 0.36573499f}, new int[]{1,2});    //block input
        INDArray out5 = Nd4j.create(new float[]{-0.17905743f, 0.19177397f}, new int[]{1,2});    //Cell state
        INDArray out6 = Nd4j.create(new float[]{-0.04025514f, 0.10104967f}, new int[]{1,2});    //Output

//        for(int i=0; i<toExec.size(); i++ ){
//            System.out.println(i + "\t" + m.get(toExec.get(i)));
//        }

        assertEquals(out0, m.get(toExec.get(0)));       //Input modulation gate
        assertEquals(out1, m.get(toExec.get(1)));       //Cell state (pre tanh)
        assertEquals(out2, m.get(toExec.get(2)));       //Forget gate
        assertEquals(out3, m.get(toExec.get(3)));       //Output gate
        assertEquals(out4, m.get(toExec.get(4)));       //block input
        assertEquals(out5, m.get(toExec.get(5)));       //Cell state
        assertEquals(out6, m.get(toExec.get(6)));       //Output
    }

    @Test
    public void testGRUCell(){
        Nd4j.getRandom().setSeed(12345);
        int mb = 2;
        int nIn = 3;
        int nOut = 4;

        SameDiff sd = SameDiff.create();
        SDVariable x = sd.constant(Nd4j.rand(DataType.FLOAT, mb, nIn));
        SDVariable hLast = sd.constant(Nd4j.rand(DataType.FLOAT, mb, nOut));
        SDVariable Wru = sd.constant(Nd4j.rand(DataType.FLOAT, (nIn+nOut), 2*nOut));
        SDVariable Wc = sd.constant(Nd4j.rand(DataType.FLOAT, (nIn+nOut), nOut));
        SDVariable bru = sd.constant(Nd4j.rand(DataType.FLOAT, 2*nOut));
        SDVariable bc = sd.constant(Nd4j.rand(DataType.FLOAT, nOut));

        double fb = 1.0;
        GRUWeights weights = GRUWeights.builder()
                .ruWeight(Wru)
                .cWeight(Wc)
                .ruBias(bru)
                .cBias(bc)
                .build();

        SDVariable[] v = sd.rnn().gruCell(x, hLast, weights);
        List<String> toExec = new ArrayList<>();
        for(SDVariable sdv : v){
            toExec.add(sdv.name());
        }

        //Test forward pass:
        Map<String,INDArray> m = sd.output(null, toExec);

        //Weights and bias order: [r, u], [c]

        //Reset gate:
        INDArray wr_x = Wru.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(0, nOut));           //Input weights
        INDArray wr_r = Wru.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(0, nOut));    //Recurrent weights
        INDArray br = bru.getArr().get(NDArrayIndex.interval(0, nOut));

        INDArray rExp = x.getArr().mmul(wr_x).addiRowVector(br);        //[mb,nIn]*[nIn, nOut] + [nOut]
        rExp.addi(hLast.getArr().mmul(wr_r));   //[mb,nOut]*[nOut,nOut]
        Transforms.sigmoid(rExp,false);

        INDArray rAct = m.get(toExec.get(0));
        assertEquals(rExp, rAct);

        //Update gate:
        INDArray wu_x = Wru.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(nOut, 2*nOut));           //Input weights
        INDArray wu_r = Wru.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(nOut, 2*nOut));    //Recurrent weights
        INDArray bu = bru.getArr().get(NDArrayIndex.interval(nOut, 2*nOut));

        INDArray uExp = x.getArr().mmul(wu_x).addiRowVector(bu);        //[mb,nIn]*[nIn, nOut] + [nOut]
        uExp.addi(hLast.getArr().mmul(wu_r));   //[mb,nOut]*[nOut,nOut]
        Transforms.sigmoid(uExp,false);

        INDArray uAct = m.get(toExec.get(1));
        assertEquals(uExp, uAct);

        //c = tanh(x * Wcx + Wcr * (hLast .* r))
        INDArray Wcx = Wc.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.all());
        INDArray Wcr = Wc.getArr().get(NDArrayIndex.interval(nIn, nIn+nOut), NDArrayIndex.all());
        INDArray cExp = x.getArr().mmul(Wcx);
        cExp.addi(hLast.getArr().mul(rExp).mmul(Wcr));
        cExp.addiRowVector(bc.getArr());
        Transforms.tanh(cExp, false);

        assertEquals(cExp, m.get(toExec.get(2)));

        //h = u * hLast + (1-u) * c
        INDArray hExp = uExp.mul(hLast.getArr()).add(uExp.rsub(1.0).mul(cExp));
        assertEquals(hExp, m.get(toExec.get(3)));
    }
}