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
        Nd4j.getExecutioner().enableVerboseMode(true);
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
                .forgetBias(1)
                .clippingCellValue(0.0)
                .build();

        List<SDVariable> v = sd.rnn().lstmBlock("lstm", conf);
        List<String> toExec = new ArrayList<>();
        for(SDVariable sdv : v){
            toExec.add(sdv.getVarName());
        }

        //Test forward pass:
        Map<String,INDArray> m = sd.exec(null, toExec);

        INDArray wz_x = W.getArr().get(NDArrayIndex.interval(0,nIn), NDArrayIndex.interval(0, nOut));           //Input weights
        INDArray wz_r = W.getArr().get(NDArrayIndex.interval(nIn,nIn+nOut), NDArrayIndex.interval(0, nOut));    //Recurrent weights
        INDArray bz = b.getArr().get(NDArrayIndex.interval(0, nOut));

        INDArray zExp = x.getArr().mmul(wz_x).addiRowVector(bz);        //[mb,nIn]*[nIn, nOut] + [nOut]
        zExp.addi(yLast.getArr().mmul(wz_r));   //[mb,nOut]*[nOut,nOut]

        INDArray zAct = m.get(toExec.get(0));
        assertEquals(zExp, zAct);

    }

}