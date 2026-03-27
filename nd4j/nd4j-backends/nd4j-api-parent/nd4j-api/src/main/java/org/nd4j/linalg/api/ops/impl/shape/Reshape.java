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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.guava.primitives.Longs;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;

@Slf4j
@NoArgsConstructor
public class Reshape extends DynamicCustomOp {

    private long[] shape;

    public final static int C_ORDER = -99;
    public final static int F_ORDER = -102;


    private transient  boolean reshapeWithViewPossible = false;
    public Reshape(SameDiff sameDiff, SDVariable i_v, long[] shape) {
        super(null, sameDiff, new SDVariable[]{i_v});
        this.shape = shape;
        //c ordering: see (char) 99 for c ordering and (char) 'f' is 102
        //note it has to be negative for the long array case only
        //to flag the difference between an ordering being specified
        //and a dimension.
        if(iArguments.isEmpty())
            addIArgument(C_ORDER);
        addIArgument(shape);
        this.reshapeWithViewPossible = org.nd4j.linalg.api.shape.Shape.ableToReshapeWithView(i_v.getArr(), iArguments.get(0) == F_ORDER, Longs.toArray(iArguments.subList(1,iArguments.size())));
    }

    public Reshape(SameDiff sameDiff, SDVariable i_v, long[] shape,char c) {
        super(null, sameDiff, new SDVariable[]{i_v});
        Preconditions.checkState(c == 'c' || c == 'f', "Invalid order: must be 'c' or 'f', got %s", c);
        this.shape = shape;
        //c ordering: see (char) 99 for c ordering and (char) 'f' is 102
        //note it has to be negative for the long array case only
        //to flag the difference between an ordering being specified
        //and a dimension.
        addIArgument(-c);
        addIArgument(shape);

    }

    public Reshape(SameDiff sameDiff, SDVariable i_v, SDVariable shape) {
        super(null, sameDiff, new SDVariable[]{i_v, shape});
        if(iArguments.isEmpty())
            addIArgument(C_ORDER);
    }

    public Reshape(INDArray in, long... shape) {
        super(new INDArray[]{in}, null);
        this.shape = shape;
        //c ordering: see (char) 99 for c ordering and (char) 'f' is 102
        //note it has to be negative for the long array case only
        //to flag the difference between an ordering being specified
        //and a dimension.
        if(iArguments.isEmpty())
            addIArgument(C_ORDER);
        addIArgument(shape);
        this.reshapeWithViewPossible = org.nd4j.linalg.api.shape.Shape.ableToReshapeWithView(in, iArguments.get(0) == F_ORDER, Longs.toArray(iArguments.subList(1,iArguments.size())));

    }


    public Reshape(INDArray in, char order,long... shape) {
        super(new INDArray[]{in}, null);
        Preconditions.checkState(order == 'c' || order == 'f', "Invalid order: must be 'c' or 'f', got %s", order);
        this.shape = shape;
        //c ordering: see (char) 99 for c ordering and (char) 'f' is 102
        //note it has to be negative for the long array case only
        //to flag the difference between an ordering being specified
        //and a dimension.
        addIArgument(-order);
        addIArgument(shape);
        this.reshapeWithViewPossible = org.nd4j.linalg.api.shape.Shape.ableToReshapeWithView(in, iArguments.get(0) == F_ORDER, Longs.toArray(iArguments.subList(1,iArguments.size())));

    }


    public Reshape(@NonNull INDArray in, @NonNull INDArray shape, INDArray out) {
        super(null, new INDArray[]{in, shape}, wrapOrNull(out), null, (List<Long>)null);
        if(iArguments.isEmpty())
            addIArgument(C_ORDER);
        this.reshapeWithViewPossible = org.nd4j.linalg.api.shape.Shape.ableToReshapeWithView(in, iArguments.get(0) == F_ORDER,shape.toLongVector());

    }

    public Reshape(INDArray in, INDArray shape) {
        this(in, shape, null);
    }



    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        if (!nodeDef.containsAttr("TShape") && nodeDef.getInputCount() == 1) {
            this.shape = new long[]{};
            return;
        } else if(nodeDef.getInputCount() == 1){
            val shape = nodeDef.getAttrOrThrow("Tshape");
            if (!shape.hasShape()) {
                val shapeRet = new long[2];
                shapeRet[0] = 1;
                shapeRet[1] = shape.getValueCase().getNumber();
                this.shape = shapeRet;
            } else {
                val shapeVals = shape.getShape().getDimList();
                if (shapeVals.size() > 1) {
                    this.shape = new long[shapeVals.size()];
                    for (int i = 0; i < shapeVals.size(); i++) {
                        this.shape[i] = (int) shapeVals.get(i).getSize();
                    }
                } else {
                    this.shape = new long[2];
                    this.shape[0] = 1;
                    this.shape[1] = (int) shapeVals.get(0).getSize();
                }

            }

            //all TF is c

            if (this.shape != null) {
                addIArgument(this.shape);
            }
        }
    }

    @Override
    public void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {

    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();

        val shapeMapping = PropertyMapping.builder()
                .onnxAttrName("shape")
                .tfInputPosition(-1)
                .propertyNames(new String[]{"shape"})
                .build();

        map.put("shape", shapeMapping);

        ret.put(tensorflowName(), map);
        ret.put(onnxName(), map);

        return ret;
    }


    @Override
    public String opName() {
        return "reshape";
    }

    @Override
    public String onnxName() {
        return "Reshape";
    }

    @Override
    public String tensorflowName() {
        return "Reshape";
    }



    @Override
    public void configureFromArguments() {
        if(iArguments.size() > 1) {
            //ordering comes first followed by the actual shape

            this.shape = new long[iArguments.size() - 1];
            for(int i = 0; i < shape.length; i++) {
                this.shape[i] = iArguments.get(i + 1);
            }

            this.reshapeWithViewPossible = org.nd4j.linalg.api.shape.Shape.ableToReshapeWithView(getInputArgument(0), iArguments.get(0) == F_ORDER, Longs.toArray(iArguments.subList(1,iArguments.size())));
        } else if(iArguments.isEmpty()) {
            iArguments.add((long) C_ORDER);
        }
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable origShape = sameDiff.shape(arg());
        SDVariable ret = sameDiff.reshape(i_v.get(0), origShape);
        return Collections.singletonList(ret);
    }
    @Override
    public boolean initializeOutputs(OpContext ctx) {
        if(!reshapeWithViewPossible)
            return super.initializeOutputs(ctx);
        else {
            char newOrder = (char) -iArguments.get(0);
            if(inputArguments.size() > 1)
                shape = inputArguments.get(1).toLongVector();

            INDArray input = inputArguments().get(0);

            // Empty input: output must also be empty. Resolve -1 to 0 and return
            // an empty array rather than trying to create a view of zero-length data.
            if (input.isEmpty() || input.length() == 0) {
                long[] resolvedShape = shape.clone();
                for (int i = 0; i < resolvedShape.length; i++) {
                    if (resolvedShape[i] == -1) {
                        resolvedShape[i] = 0;
                    }
                }
                addOutputArgument(Nd4j.emptyWithShape(resolvedShape, input.dataType()));
                return false;
            }

            // Resolve -1 dimension from actual input length
            long[] resolvedShape = shape.clone();
            int minusOneIdx = -1;
            long knownElements = 1;
            for (int i = 0; i < resolvedShape.length; i++) {
                if (resolvedShape[i] == -1) {
                    minusOneIdx = i;
                } else {
                    knownElements *= resolvedShape[i];
                }
            }
            if (minusOneIdx >= 0 && knownElements > 0) {
                resolvedShape[minusOneIdx] = input.length() / knownElements;
            }

            //wrap an existing buffer to ensure that the original buffer doesn't get deallocated
            INDArray arr = Nd4j.create(input.data(),
                    resolvedShape,
                    Nd4j.getStrides(resolvedShape, newOrder), input.offset(), newOrder);
            addOutputArgument(arr);
            return false;
        }
    }
    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Output type is always same as input type
        return Collections.singletonList(dataTypes.get(0));
    }

    @Override
    public boolean outputShapeDependsOnInputData() {
        return true;
    }

    /**
     * Calculate reshape output shape in Java to avoid JNI overhead.
     * The target shape comes from either iArgs or the second input tensor.
     * Handles -1 dimension inference (at most one -1 allowed).
     */
    @Override
    public List<DataBuffer> calculateOutputShapeFromInputs(OpContext oc) {
        if (oc == null || oc.numInputArguments() < 1) {
            return null;
        }

        INDArray input = oc.getInputArray(0);
        if (input == null) {
            return null;
        }

        long[] inputShape = input.shape();
        long inputElements = 1;
        for (long d : inputShape) {
            inputElements *= d;
        }

        // Get target shape from iArgs (skip first arg which is ordering)
        // or from second input tensor
        long[] targetShape = null;
        List<Long> iArgs = oc.getIArguments();
        if (iArgs != null && iArgs.size() > 1) {
            // iArgs[0] is ordering, iArgs[1:] is shape
            targetShape = new long[iArgs.size() - 1];
            for (int i = 1; i < iArgs.size(); i++) {
                targetShape[i - 1] = iArgs.get(i);
            }
        } else if (oc.numInputArguments() >= 2) {
            // Second input is the shape tensor
            INDArray shapeInput = oc.getInputArray(1);
            if (shapeInput != null && !shapeInput.isEmpty() && shapeInput.data() != null && !shapeInput.data().wasClosed()) {
                targetShape = shapeInput.toLongVector();
            }
        } else if (this.shape != null) {
            targetShape = this.shape;
        }

        if (targetShape == null || targetShape.length == 0) {
            return null; // Fall back to C++
        }

        // Handle -1 dimension inference
        int minusOneIdx = -1;
        long knownElements = 1;
        for (int i = 0; i < targetShape.length; i++) {
            if (targetShape[i] == -1) {
                if (minusOneIdx >= 0) {
                    return null; // Multiple -1s, fall back to C++
                }
                minusOneIdx = i;
            } else if (targetShape[i] == 0) {
                // ONNX semantics: 0 means "keep the same as input"
                if (i < inputShape.length) {
                    targetShape[i] = inputShape[i];
                }
                knownElements *= targetShape[i];
            } else {
                knownElements *= targetShape[i];
            }
        }

        long[] outputShape = targetShape.clone();
        if (minusOneIdx >= 0) {
            if (knownElements == 0) {
                return null; // Cannot infer with zero elements
            }
            outputShape[minusOneIdx] = inputElements / knownElements;
        }

        // Validate total elements match
        long outputElements = 1;
        for (long d : outputShape) {
            outputElements *= d;
        }
        if (outputElements != inputElements) {
            return null; // Element count mismatch, fall back to C++
        }

        DataType dtype = input.dataType();
        long[] strides = Nd4j.getStrides(outputShape, 'c');
        boolean isEmpty = false;
        for (long dim : outputShape) {
            if (dim == 0) { isEmpty = true; break; }
        }
        LongShapeDescriptor descriptor = LongShapeDescriptor.fromShape(outputShape, strides, 1, 'c', dtype, isEmpty);
        DataBuffer shapeInfo = Shape.createShapeInformation(descriptor);
        return Collections.singletonList(shapeInfo);
    }
}
