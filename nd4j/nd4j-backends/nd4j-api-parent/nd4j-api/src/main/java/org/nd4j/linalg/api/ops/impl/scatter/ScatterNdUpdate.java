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

package org.nd4j.linalg.api.ops.impl.scatter;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class ScatterNdUpdate extends DynamicCustomOp {
    private boolean lock = false;

    public ScatterNdUpdate(SameDiff sameDiff, SDVariable ref, SDVariable indices, SDVariable updates) {
        super(null, sameDiff, new SDVariable[]{ref, indices, updates}, false);
    }

    public ScatterNdUpdate(){}

    public ScatterNdUpdate(INDArray ref, INDArray indices, INDArray updates) {
        super(null,new INDArray[]{ref,indices,updates},null);
    }

    @Override
    public String opName() {
        return "scatter_nd_update";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "ScatterNdUpdate";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> gradOut){
        //out = ref with each index row's slice replaced by its update. scatter_nd_update applies the rows in order, so
        //where a row repeats, out holds the last of its updates. dL/dref is dL/dOut with the replaced slices zeroed;
        //an update's gradient is dL/dOut at its row for the update written last there, and 0 for the ones it replaced.
        SDVariable ref = arg(0);
        SDVariable indices = arg(1);
        SDVariable updates = arg(2);
        SDVariable grad = gradOut.get(0);

        SDVariable gradRef = sameDiff.scatterNdUpdate(grad, indices, sameDiff.zerosLike(updates));

        //Each update element numbered by its row: the number left at an output element once the rows are applied in
        //order is the row written last there.
        SDVariable rows = sameDiff.size(indices).div(sameDiff.sizeAt(indices, -1));
        SDVariable sliceLength = sameDiff.size(updates).div(rows);
        SDVariable elements = sameDiff.range(sameDiff.constant(Nd4j.scalar(0L)), sameDiff.size(updates),
                sameDiff.constant(Nd4j.scalar(1L)), DataType.INT64);
        SDVariable row = sameDiff.reshape(elements.div(sliceLength), sameDiff.shape(updates));
        SDVariable lastRow = sameDiff.scatterNdUpdate(sameDiff.fill(sameDiff.shape(ref), DataType.INT64, -1),
                indices, row);
        SDVariable written = sameDiff.gatherNd(lastRow, indices).eq(row).castTo(updates.dataType());

        return Arrays.asList(gradRef, sameDiff.zerosLike(indices), sameDiff.gatherNd(grad, indices).mul(written));
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        throw new UnsupportedOperationException("Use the new Tensorflow Importer instead. This method is now removed.");

    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes) {
        Preconditions.checkState(inputDataTypes != null && inputDataTypes.size() == 3, "Expected exactly 3 input datatypes for %s, got %s", getClass(), inputDataTypes);
        Preconditions.checkState(inputDataTypes.get(0) == inputDataTypes.get(2), "Reference (input 0) and updates (input 2) must have exactly same data types, got %s and %s",
                inputDataTypes.get(0), inputDataTypes.get(2));
        return Collections.singletonList(inputDataTypes.get(0));
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
        addBArgument(lock);
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new HashMap<>();
        ret.put("lock", lock);
        return ret;
    }



}
