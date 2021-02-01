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

package org.nd4j.autodiff.util;

import java.util.*;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.ExternalErrorsFunction;
import org.nd4j.linalg.api.ops.impl.shape.ReductionShape;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JException;
import org.nd4j.linalg.factory.Nd4j;

@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class SameDiffUtils {

    /**
     * Stack batch outputs, like an output from {@link org.nd4j.autodiff.samediff.SameDiff#output(MultiDataSetIterator, String...)}
     */
    public static Map<String, INDArray> stackOutputs(List<Map<String, INDArray>> outputs){
        Map<String, List<INDArray>> outs = new HashMap<>();
        for(Map<String, INDArray> batch : outputs){
            for(String k : batch.keySet()){
                if(!outs.containsKey(k))
                    outs.put(k, new ArrayList<INDArray>());
                outs.get(k).add(batch.get(k));
            }
        }

        Map<String, INDArray> ret = new HashMap<>();
        for(String k : outs.keySet()){
            try {
                ret.put(k, Nd4j.concat(0, outs.get(k).toArray(new INDArray[0])));
            } catch(Exception e){
                throw new ND4JException("Error concatenating batch outputs", e);
            }
        }
        return ret;
    }

    /**
     * Get a list of batch outputs for a single variable from a list of batch outputs for all variables
     */
    public static List<INDArray> getSingleOutput(List<Map<String, INDArray>> outputs, String output){
        List<INDArray> batches = new ArrayList<>();
        for(Map<String, INDArray> batch : outputs)
            batches.add(batch.get(output));

        return batches;
    }

    public static ExternalErrorsFunction externalErrors(SameDiff sameDiff, Map<String, INDArray> externalGradients, SDVariable... inputs) {
        Preconditions.checkArgument(inputs != null && inputs.length > 0, "Require at least one SDVariable to" +
                " be specified when using external errors: got %s", inputs);
        ExternalErrorsFunction fn = new ExternalErrorsFunction(sameDiff, Arrays.asList(inputs), externalGradients);
        fn.outputVariable();
        return fn;
    }

    public static ExternalErrorsFunction externalErrors(SameDiff sameDiff, SDVariable[] inputs) {
        return externalErrors(sameDiff, null, inputs);
    }



    /**
     * Add 1s as required to the array make an array possible to be broadcast with the original (pre-reduce) array.
     * <p>
     * Example: if doing [a,b,c].sum(1), result is [a,c]. To 'undo' this in a way that can be auto-broadcast,
     * we want to expand as required - i.e., [a,c] -> [a,1,c] which can be auto-broadcast with the original [a,b,c].
     * This is typically only used with reduction operations backprop.
     *
     * @param origRank   Rank of the original array, before the reduction was executed
     * @param reduceDims Dimensions that the original array was reduced from
     * @param toExpand   Array to add 1s to the shape to (such that it can be
     * @return Reshaped array.
     */
    public static SDVariable reductionBroadcastableWithOrigShape(int origRank, int[] reduceDims, SDVariable toExpand) {
        if (Shape.isWholeArray(origRank, reduceDims)) {
            //Output is [1,1] which is already broadcastable
            return toExpand;
        } else if (origRank == 2 && reduceDims.length == 1) {
            //In this case: [a,b] -> [1,b] or [a,b] -> [a,1]
            //both are already broadcastable
            return toExpand;
        } else {
            //Example: [a,b,c].sum(1) -> [a,c]... want [a,1,c]
            for (int d : reduceDims) {
                toExpand = toExpand.getSameDiff().expandDims(toExpand, d);
            }
            return toExpand;
        }
    }

    public static SDVariable reductionBroadcastableWithOrigShape(SDVariable origInput, SDVariable axis, SDVariable toExpand) {
        SDVariable shape = origInput.shape();
        SDVariable reduceShape = reductionShape(shape, axis, true);
        SDVariable reshaped = toExpand.reshape(reduceShape);
        return reshaped;
    }

    public static SDVariable reductionShape(SDVariable shape, SDVariable axis, boolean keepDim){
        return new ReductionShape(shape.getSameDiff(), shape, axis, keepDim).outputVariable();
    }

    public static void validateDifferentialFunctionSameDiff(SameDiff sameDiff, SDVariable function, DifferentialFunction op) {

        Preconditions.checkState(function != null, "Passed in function was null.");
        Preconditions.checkState(function.getSameDiff() == sameDiff);

        Preconditions.checkState(function.getSameDiff() == sameDiff,
                "Function applications must be contained " +
                        "in same sameDiff. The left %s must match this function %s", function, op);
    }
}
