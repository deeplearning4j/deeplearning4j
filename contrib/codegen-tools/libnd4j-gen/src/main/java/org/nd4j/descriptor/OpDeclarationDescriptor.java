/*******************************************************************************
 * Copyright (c) 2020 Konduit KK.
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
package org.nd4j.descriptor;

import lombok.Builder;
import lombok.Data;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * The op descriptor for the libnd4j code base.
 * Each op represents a serialized version of
 * {@link org.nd4j.linalg.api.ops.DynamicCustomOp}
 * that has naming metadata attached.
 *
 * @author Adam Gibson
 */
@Data
@Builder(toBuilder = true)
public class OpDeclarationDescriptor implements Serializable  {
    private String name;
    private int nIn,nOut,tArgs,iArgs;
    private boolean inplaceAble;
    private List<String> inArgNames;
    private List<String> outArgNames;
    private List<String> tArgNames;
    private List<String> iArgNames;
    private List<String> bArgNames;


    private OpDeclarationType opDeclarationType;
    @Builder.Default
    private Map<String,Boolean> argOptional = new HashMap<>();


    public enum OpDeclarationType {
        CUSTOM_OP_IMPL,
        BOOLEAN_OP_IMPL,
        LIST_OP_IMPL,
        LOGIC_OP_IMPL,
        OP_IMPL,
        DIVERGENT_OP_IMPL,
        CONFIGURABLE_OP_IMPL,
        REDUCTION_OP_IMPL,
        BROADCASTABLE_OP_IMPL,
        BROADCASTABLE_BOOL_OP_IMPL,
        LEGACY_XYZ,
        PLATFORM_IMPL
    }



    public void validate() {
        if(nIn >= 0 && nIn != inArgNames.size() && !isVariableInputSize()) {
            System.err.println("In arg names was not equal to number of inputs found for op " + name);
        }

        if(nOut >= 0 && nOut != outArgNames.size() && !isVariableOutputSize()) {
            System.err.println("Output arg names was not equal to number of outputs found for op " + name);
        }

        if(tArgs >= 0 && tArgs != tArgNames.size() && !isVariableTArgs()) {
            System.err.println("T arg names was not equal to number of T found for op " + name);
        }
        if(iArgs >= 0 && iArgs != iArgNames.size() && !isVariableIntArgs()) {
            System.err.println("Integer arg names was not equal to number of integer args found for op " + name);
        }
    }


    /**
     * Returns true if there is a variable number
     * of integer arguments for an op
     * @return
     */
    public boolean isVariableIntArgs() {
        return iArgs < 0;
    }

    /**
     * Returns true if there is a variable
     * number of t arguments for an op
     * @return
     */
    public boolean isVariableTArgs() {
        return tArgs < 0;
    }

    /**
     * Returns true if the number of outputs is variable size
     * @return
     */
    public boolean isVariableOutputSize() {
        return nOut < 0;
    }

    /**
     * Returns true if the number of
     * inputs is variable size
     * @return
     */
    public boolean isVariableInputSize() {
        return nIn < 0;
    }


}
