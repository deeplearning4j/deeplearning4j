/*-
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
 *
 */

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * AlphaDropOut implementation as Op
 *
 * @author raver119@gmail.com
 */
public class AlphaDropOut extends BaseRandomOp {

    private double p;
    private double a;
    private double alphaPrime;
    private double b;

    public AlphaDropOut() {

    }

    public AlphaDropOut(@NonNull INDArray x, double p, double alpha, double alphaPrime, double beta) {
        this(x, x, p, alpha, alphaPrime, beta, x.lengthLong());
    }

    public AlphaDropOut(@NonNull INDArray x, @NonNull INDArray z, double p, double alpha, double alphaPrime, double beta) {
        this(x, z, p, alpha, alphaPrime, beta, x.lengthLong());
    }

    public AlphaDropOut(@NonNull INDArray x, @NonNull INDArray z, double p, double alpha, double alphaPrime, double beta, long n) {
        this.p = p;
        this.a = alpha;
        this.b = beta;
        this.alphaPrime = alphaPrime;
        init(x, null, z, n);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new LinkedHashMap<>();
        ret.put("p",p);
        ret.put("a",a);
        ret.put("alphaPrime",alphaPrime);
        ret.put("b",b);
        return ret;
    }

    @Override
    public int opNum() {
        return 12;
    }

    @Override
    public String opName() {
        return "alpha_dropout";
    }

    @Override
    public void init(INDArray x, INDArray y, INDArray z, long n) {
        super.init(x, y, z, n);
        this.extraArgs = new Object[] {p, a, b, alphaPrime};
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}
