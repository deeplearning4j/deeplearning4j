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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * DeConv3DDerivative operation
 */
@Slf4j
public class DeConv3DDerivative extends DynamicCustomOp {

    protected DeConv3DConfig config;

    public DeConv3DDerivative() {}

    public DeConv3DDerivative(SameDiff sameDiff, @NonNull SDVariable input, @NonNull SDVariable weights, SDVariable bias, SDVariable grad, DeConv3DConfig config) {
        super(sameDiff, toArr(input, weights, bias, grad));
        this.config = config;
        addArgs();
    }

    private static SDVariable[] toArr(SDVariable input, SDVariable weights, SDVariable bias, SDVariable grad){
        if(bias != null){
            return new SDVariable[]{input, weights, bias, grad};
        } else {
            return new SDVariable[]{input, weights, grad};
        }
    }

    @Override
    public String opName() {
        return "deconv3d_bp";
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        if(config == null && !iArguments.isEmpty()){
            config = DeConv3DConfig.builder()
                    .kD(iArguments.get(0))
                    .kH(iArguments.get(1))
                    .kW(iArguments.get(2))
                    .sD(iArguments.get(3))
                    .sH(iArguments.get(4))
                    .sW(iArguments.get(5))
                    .pD(iArguments.get(6))
                    .pH(iArguments.get(7))
                    .pW(iArguments.get(8))
                    .dD(iArguments.get(9))
                    .dH(iArguments.get(10))
                    .dW(iArguments.get(11))
                    .isSameMode(iArguments.get(12) == 1)
                    .dataFormat(iArguments.get(13) == 1 ? DeConv3DConfig.NDHWC : DeConv3DConfig.NCDHW)
                    .build();
        }
        return config.toProperties();
    }

    private void addArgs() {
        addIArgument(config.getKD());
        addIArgument(config.getKH());
        addIArgument(config.getKW());
        addIArgument(config.getSD());
        addIArgument(config.getSH());
        addIArgument(config.getSW());
        addIArgument(config.getPD());
        addIArgument(config.getPH());
        addIArgument(config.getPW());
        addIArgument(config.getDD());
        addIArgument(config.getDH());
        addIArgument(config.getDW());
        addIArgument(ArrayUtil.fromBoolean(config.isSameMode()));
        addIArgument(config.getDataFormat().equalsIgnoreCase(DeConv3DConfig.NCDHW) ? 0 : 1);
    }


    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No op name found for backwards.");
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No op name found for backwards");
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        throw new UnsupportedOperationException("Gradient of DeConv3DDerivative not supported.");

    }

    @Override
    public int getNumOutputs(){
        //Inputs: in, weights, optional bias, gradOut                      3 req, 1 optional
        //Outputs: gradAtInput, gradW, optional gradB                      2 req, 1 optional
        SDVariable[] args = args();
        return args.length - 1;
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        int n = args().length;  //Original inputs + gradient at
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        List<DataType> out = new ArrayList<>(n-1);
        for( int i=0; i<n-1; i++ ){
            out.add(inputDataTypes.get(i));
        }
        return out;
    }
}
