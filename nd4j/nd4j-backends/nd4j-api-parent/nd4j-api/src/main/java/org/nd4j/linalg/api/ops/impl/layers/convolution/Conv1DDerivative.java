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

package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.common.util.ArrayUtil;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


@Slf4j
@Getter
@NoArgsConstructor
public class Conv1DDerivative extends DynamicCustomOp {

    protected Conv1DConfig config;
    private static final String INVALID_CONFIGURATION = "Invalid Conv1D configuration : s = %s p = %s ";

    public Conv1DDerivative(@NonNull SameDiff sameDiff,
                            @NonNull SDVariable[] inputs,
                            @NonNull Conv1DConfig config) {
        super(sameDiff, inputs);
        initConfig(config);
    }

    public Conv1DDerivative(@NonNull SameDiff sd, @NonNull SDVariable input, @NonNull SDVariable weights, SDVariable bias, SDVariable gradOut, @NonNull Conv1DConfig config){
        this(sd, wrapFilterNull(input, weights, bias, gradOut), config);
    }

    public Conv1DDerivative(INDArray[] inputs, INDArray[] outputs, Conv1DConfig config) {
        super(inputs, outputs);

        initConfig(config);
    }

    public Conv1DDerivative(@NonNull INDArray input, @NonNull INDArray weights, INDArray bias, @NonNull INDArray gradOut, INDArray output, @NonNull Conv1DConfig config){
        this(wrapFilterNull(input, weights, bias, gradOut), wrapOrNull(output), config);
    }

    private void initConfig(Conv1DConfig config){
        this.config = config;
        Preconditions.checkState(config.getS() >= 1 && config.getP() >= 0, INVALID_CONFIGURATION, config.getS(), config.getP());
        addArgs();
    }

    protected void addArgs() {
        if (config == null)
            config = Conv1DConfig.builder().build();

        addIArgument(config.getK(),
                config.getS(),
                config.getP(),
                config.getD(),
                config.getPaddingMode().ordinal(),
                ArrayUtil.fromBoolean(config.isNWC()));
    }

    @Override
    public long[] iArgs() {
        if (iArguments.size() == 0)
            addArgs();

        return super.iArgs();
    }

    @Override
    public Object getValue(Field property) {
        if (config == null && !iArguments.isEmpty()) {
            config = Conv1DConfig.builder()
                    .k(iArguments.get(0))
                    .s(iArguments.get(1))
                    .p(iArguments.get(2))
                    .d(iArguments.get(3))
                    .paddingMode(PaddingMode.values()[iArguments.get(4).intValue()])
                    .dataFormat(iArguments.get(5) == 1 ? Conv1DConfig.NCW : Conv1DConfig.NWC)
                    .build();
        }

        return config.getValue(property);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        return config.toProperties();
    }

    @Override
    public boolean isConfigProperties() {
        return true;
    }

    @Override
    public String configFieldName() {
        return "config";
    }

    @Override
    public String opName() {
        return "conv1d_bp";
    }

    @Override
    public int getNumOutputs() {
        if(args().length == 4){
            return 3;   //Includes bias
        } else {
            return 2;   //No bias - only input + weight grads
        }
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        int n = args().length;
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == n, "Expected %s input data types for %s, got %s", n, getClass(), inputDataTypes);
        return new ArrayList<>(inputDataTypes.subList(0, inputDataTypes.size()-1)); //All except gradient input variable
    }
}
