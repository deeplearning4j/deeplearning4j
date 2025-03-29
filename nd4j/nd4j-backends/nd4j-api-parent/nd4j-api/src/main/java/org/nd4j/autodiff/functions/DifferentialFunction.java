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

package org.nd4j.autodiff.functions;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.Onnx;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.serde.FlatBuffersMapper;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.util.StackTraceUtils;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;


@Data
@Slf4j
public abstract class DifferentialFunction {

    @Getter
    @Setter
    @JsonIgnore
    protected SameDiff sameDiff;

    @Getter
    @Setter
    @JsonIgnore
    protected boolean inPlace;



    @Getter
    @Setter
    @JsonIgnore
    protected INDArray scalarValue;


    @Getter
    @Setter
    @JsonIgnore
    protected long[] dimensions;

    @JsonIgnore
    protected Object[] extraArgs;


    @Getter
    @Setter
    @JsonIgnore
    protected String ownName;

    @JsonIgnore
    @Getter
    @Setter
    protected boolean ownNameSetWithDefault = false;

    @Getter
    protected StackTraceElement creationLocation,creationPointofOrigin;
    @Getter
    protected StackTraceElement[] sameDiffCalls;
    @Getter
    protected  StackTraceElement[] creationCallStack;
    public DifferentialFunction() {
        this(false);
    }

    public DifferentialFunction(boolean sameDiff) {
        //Only need instance ID if using function in context of SameDiff, not standard ND4J with INDArray args
        if(sameDiff) {
            setInstanceId();
        }

        recordCreation();

    }

    /**
     * Initialize the function from the given
     * {@link NodeDef}
     * @param nodeDef
     */
    public DifferentialFunction(SameDiff sameDiff,NodeDef nodeDef, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        this.sameDiff = sameDiff;
        setInstanceId();
        initFromTensorFlow(nodeDef, sameDiff,attributesForNode ,graph);
        recordCreation();
    }

    /**
     * Initialize the function from the given
     * {@link Onnx.NodeProto}
     * @param node
     */
    public DifferentialFunction(SameDiff sameDiff, Onnx.NodeProto node, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph) {
        this.sameDiff = sameDiff;
        setInstanceId();
        initFromOnnx(node, sameDiff, attributesForNode, graph);
        recordCreation();
    }


    public String debugInfo() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("Op type: " + opName());
        if(getOwnName() != null) {
            stringBuilder.append("Own name: " + getOwnName());
        }

        if(sameDiff != null) {
            String[] inputsForOp = sameDiff.getInputsForOp(this);
            if(inputsForOp != null) {
                stringBuilder.append("Input names: " + Arrays.toString(inputsForOp) + "\n");
                for(String variable : inputsForOp) {
                    SDVariable var = sameDiff.getVariable(variable);
                    stringBuilder.append(var.toString() + "\n");
                }
            }

            String[] outputsForOp = sameDiff.getOutputsForOp(this);
            if(outputsForOp != null) {
                stringBuilder.append("Output names: " + Arrays.toString(outputsForOp) + "\n");
                for(String output : outputsForOp) {
                    SDVariable outVar = sameDiff.getVariable(output);
                    stringBuilder.append(outVar.toString() + "\n");
                }
            }
        }


        return stringBuilder.toString();


    }



    protected void recordCreation() {
        if(Nd4j.getEnvironment().isDebug() || Nd4j.getEnvironment().isVerbose()) {
            StackTraceElement[] stackTrace = Thread.currentThread().getStackTrace();
            this.creationLocation = StackTraceUtils.pointOfInvocation(stackTrace);
            this.creationPointofOrigin = StackTraceUtils.pointOfOrigin(stackTrace);
            this.sameDiffCalls = StackTraceUtils.callsFromClass(stackTrace, SameDiff.class.getName());
            creationCallStack = stackTrace;
        }
    }

    /**
     * Returns the {@link AttributeAdapter} s for each of the
     * possible ops for import (typically tensorflow and onnx)
     *
     * See {@link AttributeAdapter} for more information on what the
     * adapter does.
     *
     * Similar to {@link #mappingsForFunction()}, the returned map
     * contains a {@link AttributeAdapter} for each field name
     * when one is present. (It is optional for one to exist)_
     * @return
     */
    public Map<String,Map<String,AttributeAdapter>> attributeAdaptersForFunction() {
        return Collections.emptyMap();
    }

    /**
     * Returns the mappings for a given function (
     * for tensorflow and onnx import mapping properties
     * of this function). The mapping is indexed by field name.
     * If the function has no properties, this returned map
     * will be empty.
     *
     * Note that some functions have multiple names.
     * This function returns a map indexed by each
     * alias it has for a given name.
     * These names include both onnx and tensorflow names (which might be 1 or more)
     *
     * @return
     */
    public Map<String,Map<String,PropertyMapping>> mappingsForFunction() {
        return Collections.emptyMap();
    }

    /**
     * Returns the properties for a given function
     * @return
     */
    public Map<String,Object> propertiesForFunction() {
        Map<String,Field> fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);
        Map<String,Object> ret = new LinkedHashMap<>();
        Preconditions.checkNotNull(fields, "DifferentialFunctionClassHolder returned null fields for %s - op has not been added to ImportClassMapping?", getClass());

        for(val entry : fields.entrySet()) {
            try {
                ret.put(entry.getKey(),fields.get(entry.getKey()).get(this));
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Unable to get property for field: " + entry.getKey(), e);
            }
        }

        return ret;
    }

    public void configureWithSameDiff(SameDiff sameDiff) {
        //no op on purpose, meant to be overridden
    }

    public void setPropertiesForFunction(Map<String,Object> properties) {
        Map<String,Field> fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);
        for(String s : properties.keySet()) {
            Field f = fields.get(s);
            if(f == null){
                log.warn("No fields found for property name {} for class {}", s, this.getClass().getName());
                continue;
            }
            setValueFor(f, properties.get(s));
        }
    }

    protected Boolean getBooleanFromProperty(String propertyName,Map<String,Object> properties) {
        if(properties.containsKey(propertyName)) {
            Boolean value = (Boolean) properties.get(propertyName);
            return value;
        }

        return null;
    }

    protected String getStringFromProperty(String propertyName,Map<String,Object> properties) {
        if(properties.containsKey(propertyName)) {
            String value = (String) properties.get(propertyName);
            return value;
        }

        return null;
    }


    protected Integer getIntValueFromProperty(String propertyName, Map<String,Object> properties) {
        if(properties.containsKey(propertyName)) {
            Number value = (Number) properties.get(propertyName);
            return value.intValue();
        }

        return null;
    }


    protected Long getLongValueFromProperty(String propertyName, Map<String,Object> properties) {
        if(properties.containsKey(propertyName)) {
            Number value = (Number) properties.get(propertyName);
            return value.longValue();
        }

        return null;
    }

    protected Double getDoubleValueFromProperty(String propertyName, Map<String,Object> properties) {
        if(properties.containsKey(propertyName)) {
            Number value = (Number) properties.get(propertyName);
            return value.doubleValue();
        }

        return null;
    }


    /**
     * Get the value for a given property
     * for this function
     * @param property the property to get
     * @return the value for the function if it exists
     */
    public Object getValue(Field property) {
        try {
            return property.get(this);
        } catch (IllegalAccessException e) {
            log.error("",e);
        }

        return null;
    }

    /**
     * Set the value for this function.
     * Note that if value is null an {@link ND4JIllegalStateException}
     * will be thrown.
     * @param target the target field
     * @param value the value to set
     */
    @SneakyThrows
    public void setValueFor(Field target, Object value) {
        if(value == null && target.getType().isPrimitive()) {
            throw new ND4JIllegalStateException("Unable to set primitive field " + target + " of type " + target.getClass()
                    + " using null value!");
        }

        if(value != null) {
            value = ensureProperType(target, value);
        }

        if(isConfigProperties()) {
            String propertyName = configFieldName();
            if(propertyName == null)
                propertyName = "config";
            Field f = null;
            Class<?> currClass = getClass();
            try{
                f = currClass.getDeclaredField(propertyName);
            } catch (NoSuchFieldException e){
                //OK, try superclass
            }
            while(f == null && currClass.getSuperclass() != null) {
                currClass = currClass.getSuperclass();
                try{
                    f = currClass.getDeclaredField(propertyName);
                } catch (NoSuchFieldException e) {
                    //OK, try superclass
                }
            }

            if(f == null){
                throw new IllegalStateException("Could not find field \"" + propertyName + "\" for class " + getClass().getName());
            }

            try {
                f.setAccessible(true);
                Object o = f.get(this);
                if(o == null){
                    //Null config class - try to create one...
                    Class<?> c = f.getType();
                    try {
                        o = c.newInstance();
                    } catch (InstantiationException e){
                        throw new RuntimeException("Error creating new instance of configuration object type " + c.getName(), e);
                    }
                    f.set(this, o);
                }
                target.set(o, value);
            } catch (IllegalAccessException e){
                throw new RuntimeException("Error setting configuration field \"" + propertyName + "\" for config field \"" + propertyName
                        + "\" on class " + getClass().getName());
            }

        } else {
            try {
                //Edge case: we store float fields as doubles, rather than introduce an extra property
                if(target.getType() == float.class && value instanceof Double) {
                    value = ((Double) value).floatValue();
                }
                //Edge case: we store char fields as integers, rather than introduce an extra property
                if(target.getType() == char.class && value instanceof Integer) {
                    value = (char)((Integer)value).intValue();
                }

                if(target.getType() == char.class && value instanceof Long){
                    value = (char)((Long)value).intValue();
                }

                if(target.getType() == int.class && value instanceof  Long) {
                    Long value2 = (Long) value;
                    value = value2.intValue();
                }

                if(target.getType().equals(Integer.class) && value instanceof Long) {
                    Long value2 = (Long) value;
                    value = value2.intValue();
                }

                if(target.getType().equals(Long.class) && value instanceof Integer) {
                    Integer value2 = (Integer) value;
                    value = value2.longValue();
                }


                if(target.getType().equals(Double.class) && value instanceof Long) {
                    Long value2 = (Long) value;
                    value = value2.doubleValue();
                }

                if(target.getType().equals(Boolean.class) || target.getType().equals(boolean.class) && value instanceof Number) {
                    Number value2 = (Number) value;
                    value = value2.doubleValue() > 0;
                }

                if(target.getType().equals(DataType.class) && value instanceof Double) {
                    Double value2 = (Double) value;
                    int idxConverted = value2.intValue();
                    value = DataType.values()[idxConverted];
                }

                if(target.getType().isEnum() && (value instanceof Long || value instanceof Integer && !target.getType().equals(int.class) && !target.getType().equals(long.class))) {
                    Class<? extends Enum> enumType = (Class<? extends Enum>) target.getType();
                    Method method = enumType.getMethod("values");
                    method.setAccessible(true);
                    Object[] invoke = (Object[])method.invoke(null);
                    Number number = (Number) value;
                    int idx = number.intValue();
                    Object get = invoke[idx];
                    value = get;
                }



                target.set(this,value);
            } catch (Exception e) {
                throw new RuntimeException("Error setting property for function " + getClass().getName(), e);
            }
        }
    }


    private Object ensureProperType(Field targetType,Object value) {
        val firstClass = targetType.getType();
        val valueType = value.getClass();

        if(!firstClass.equals(valueType)) {
            if(firstClass.isEnum()){
                if(valueType.equals(String.class)) {
                    Object[] enumConstants = firstClass.getEnumConstants();
                    for (int i = 0; i < enumConstants.length; i++) {
                        if (enumConstants[i].toString().equalsIgnoreCase((String) value)) {
                            return enumConstants[i];
                        }
                    }
                    throw new IllegalStateException("Could not find enum constant value for value \"" + value
                            + "\" for enum class " + firstClass.getName());
                }
            } else if(firstClass.equals(int[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.intValue();
                }

                int otherValue = (int) value;
                int[] setValue = new int[] {otherValue};
                return setValue;
            }
            else if(firstClass.equals(Integer[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.intValue();
                }

                Integer otherValue = (Integer) value;
                Integer[] setValue = new Integer[] {otherValue};
                return setValue;
            }
            else if(firstClass.equals(long[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.longValue();
                }

                long otherValue = (long) value;
                long[] setValue = new long[] {otherValue};
                return setValue;

            }
            else if(firstClass.equals(Long[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.longValue();
                }

                Long otherValue = (Long) value;
                Long[] setValue = new Long[] {otherValue};
                return setValue;

            }
            else if(firstClass.equals(double[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.doubleValue();
                }


                double otherValue = (double) value;
                double[] setValue = new double[] {otherValue};
                return setValue;

            }
            else if(firstClass.equals(Double[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.doubleValue();
                }


                Double otherValue = (Double) value;
                Double[] setValue = new Double[] {otherValue};
                return setValue;

            }
            else if(firstClass.equals(float[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.floatValue();
                }


                float otherValue = (float) value;
                float[] setValue = new float[] {otherValue};
                return setValue;

            }
            else if(firstClass.equals(Float[].class)) {
                if(value instanceof Number) {
                    Number number = (Number) value;
                    value = number.floatValue();
                }



                Float otherValue = (Float) value;
                Float[] setValue = new Float[] {otherValue};
                return setValue;

            }
        }

        return value;
    }


    /**
     * Returns true if the fields for this class should be looked up from a configuration class.
     * @return
     */
    public boolean isConfigProperties() {
        return false;
    }

    /**
     * Returns the name of the field to be used for looking up field names.
     * This should be used in conjunction with {@link #isConfigProperties()}
     *  to facilitate mapping fields for model import.
     * @return
     */
    public String configFieldName() {
        return null;
    }


    /**
     *
     * @param sameDiff
     * @param extraArgs
     */
    public DifferentialFunction(SameDiff sameDiff,boolean inPlace, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        setInstanceId();
        this.extraArgs = extraArgs;
    }


    /**
     *
     * @param sameDiff
     * @param extraArgs
     */
    public DifferentialFunction(SameDiff sameDiff, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        setInstanceId();
        this.extraArgs = extraArgs;
    }

    /**
     *
     * @param sameDiff
     * @param args
     */
    public DifferentialFunction(SameDiff sameDiff, SDVariable[] args) {
        this(sameDiff,false, args);
    }


    /**
     * Add the various arguments for
     * this function
     * @param sameDiff
     * @param inPlace
     * @param args
     */
    public DifferentialFunction(SameDiff sameDiff, boolean inPlace, SDVariable[] args) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        setInstanceId();
        if(sameDiff != null && args != null) {
            sameDiff.addArgsFor(args, this);
        }

        recordCreation();
    }

    /**
     * Replace argument at the specified index
     * @param i the index
     * @param newArg the new argument
     */
    public void replaceArg(int i, SDVariable newArg) {
        if(sameDiff != null){
            sameDiff.replaceArgFor(i, newArg, this);
        }
    }


    /**
     * Return the output variables for this differential function.
     * Note that this op *may* dynamically generate variable outputs.
     * @return
     */
    public  SDVariable[] outputVariables() {
        return outputVariables(getOwnName() != null ? getOwnName() : opName());
    }

    /**
     * @return The output variable, or the first output variable, if multiple outputs exist
     */
    public SDVariable outputVariable() {
        return outputVariables()[0];
    }

    public List<SDVariable> outputs() {
        SDVariable[] out = outputVariables();
        return out == null ? null : Arrays.asList(out);
    }


    public String[] outputVariablesNames() {
        SDVariable[] outputVars = outputVariables();
        if(outputVars == null)
            return new String[0];
        String[] out = new String[outputVars.length];
        for( int i = 0; i < out.length; i++) {
            out[i] = outputVars[i] == null ? "" : outputVars[i].name();
        }
        return out;
    }


    /**
     * Return the output functions for this differential function.
     * @return
     */
    public abstract SDVariable[] outputVariables(String baseName);



    /**
     * The actual implementation for automatic differentiation.
     *
     * @param f1
     * @return
     */
    public abstract List<SDVariable> doDiff(List<SDVariable> f1);


    /**
     * Return the arguments for a given function
     * @return the arguments for a given function
     */
    public  SDVariable[] args() {
        return sameDiff == null ? null : sameDiff.getInputVariablesForOp(this);
    }

    /**
     * Return the variables expecting
     * gradients. This is usually {@link #args()}
     * but may vary depending on the function.
     * @return the variables expecting a gradient.
     */
    public  SDVariable[] variablesExpectingGrads() {
        return args();
    }

    /**
     * Return the specified argument for this function
     * @param num Number of the argument. Must be in range 0 to numArgs - 1 inclusive
     * @return Specified argument
     */
    public SDVariable arg(int num) {
        SDVariable[] args = args();
        Preconditions.checkNotNull(args, "Arguments are null for function %s", this.getOwnName());
        Preconditions.checkArgument(num >= 0 && num < args.length, "Invalid index: must be 0 to numArgs (0 <= idx < %s), got %s", args.length, num);
        return args[num];
    }

    public String[] argNames() {
        SDVariable[] args = args();
        if(args == null)
            return new String[0];
        String[] out = new String[args.length];
        for( int i = 0; i < args.length; i++) {
            out[i] = args[i].name();
        }
        return out;
    }

    /**
     * Return the first argument
     * @return
     */
    public SDVariable arg() {
        if(args() == null || args().length == 0)
            return null;
        return args()[0];
    }


    /**
     * Perform automatic differentiation
     * wrt the input variables
     * @param i_v1 the input variables
     * @return the differentiated output
     * wrt each input variable
     */
    public List<SDVariable> diff(List<SDVariable> i_v1) {
        List<SDVariable> vals = doDiff(i_v1);
        if(vals == null) {
            throw new IllegalStateException("Error executing diff operation: doDiff returned null for op: " + this.opName());
        }

        val outputVars = variablesExpectingGrads();
        boolean copied = false;
        for(int i = 0; i < vals.size(); i++) {
            SDVariable var = outputVars[i];
            SDVariable grad = var.hasGradient() ? var.getGradient() : null;
            if(grad != null) {
                if(!copied) {
                    //Don't mutate the original - this could mess with the original op's state!
                    vals = new ArrayList<>(vals);
                    copied = true;
                }

                SDVariable gradVar =  var.getSameDiff().math.add(grad, vals.get(i));
                vals.set(i, gradVar);
                sameDiff.setGradientForVariableName(var.name(), gradVar);
            } else {
                SDVariable gradVar = vals.get(i);
                if(sameDiff.hasVariable(var.name() + "-grad")) {
                    if(sameDiff.getVariable(var.name() + "-grad").dataType().isFPType())
                        sameDiff.getVariable(var.name() + "-grad").add(gradVar);
                } else {
                    sameDiff.updateVariableNameAndReference(gradVar,var.name() + "-grad");
                    sameDiff.setGradientForVariableName(var.name(), gradVar);
                }


            }
        }

        return vals;
    }


    /**
     * Note: DO NOT USE THIS METHOD UNLESS YOU KNOW WHAT YOU ARE DOING.
     * This is only for usage in {@link SameDiff#dynamic(String, List, List, List, List, List, List)}
     *
     */
    public void setInstanceId() {
        if(ownName == null) {
            ownNameSetWithDefault = true;
            if(sameDiff == null)
                this.ownName = UUID.randomUUID().toString();
            else {
                String n = sameDiff.getOpName(opName());
                this.ownName = n;
            }

            if(sameDiff != null)
                sameDiff.putOpForId(ownName,this);
        }
    }


    /**
     * The name of the op
     * @return
     */
    public String opName() {
        throw new UnsupportedOperationException();
    }


    /**
     * The type of the op
     * @return
     */
    public Op.Type opType() {
        throw new UnsupportedOperationException();
    }


    /**
     * The number of the op (mainly for old legacy XYZ ops
     * like {@link Op})
     * @return
     */
    public int opNum() {
        throw new UnsupportedOperationException();
    }

    @JsonIgnore
    public INDArray getInputArgument(int index){
        //Subclasses should implement this
        throw new UnsupportedOperationException("Not implemented");
    }



    /**
     * Initialize the function from the given
     * {@link NodeDef}
     * @param nodeDef
     * @param initWith
     * @param attributesForNode
     * @param graph
     */
    public abstract void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph);

    /**
     * Iniitialize the function from the given
     * {@link Onnx.NodeProto}
     * @param node
     * @param initWith
     * @param attributesForNode
     * @param graph
     */
    public abstract void initFromOnnx(Onnx.NodeProto node, SameDiff initWith, Map<String, Onnx.AttributeProto> attributesForNode, Onnx.GraphProto graph);



    /**
     * The left argument for this function
     * @return
     */
    public SDVariable larg() {
        val args = args();
        if(args == null || args.length == 0)
            throw new ND4JIllegalStateException("No arguments found.");
        return args()[0];
    }

    /**
     * The right argument for this function.
     * Note that this assumes that there are 2 args for this
     * function, if 2 are not set, it throws an
     * {@link ND4JIllegalStateException}
     * @return
     */
    public SDVariable rarg() {
        val args = args();
        if(args == null || args.length != 2)
            throw new ND4JIllegalStateException("In order to use this function, the number of arguments for this function must be 2.");
        return args[1];
    }


    /**
     * Duplicate this function
     * @return
     */
    public DifferentialFunction dup() {
        return FlatBuffersMapper.cloneViaSerialize(sameDiff, this);
    }

    /**
     * Calculate the output shape for this op
     *
     * @return List of output shape descriptors
     */
    public List<DataBuffer> calculateOutputShape() {
        throw new ND4JIllegalStateException("Op type of " + getClass().getName() + "did not override calculateOutputShape() method leaked out for [" + this.opName() + "]");
    }

    public List<DataBuffer> calculateOutputShape(OpContext oc){
        throw new ND4JIllegalStateException("Op type of " + getClass().getName() + " did not override calculateOutputShape(OpContext) method leaked out for [" + this.opName() + "]");
    }

    /**
     * Calculate the data types for the output arrays.
     * Though datatypes can also be inferred from {@link #calculateOutputShape()}, this method differs in that it does not
     * require the input arrays to be populated.
     * This is important as it allows us to do greedy datatype inference for the entire net - even if arrays are not
     * available.
     *
     * @param dataTypes The data types of the inputs
     * @return The data types of the outputs
     */
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes){
        throw new UnsupportedOperationException("Op type of " + getClass().getName() + " and name " +  this.toString() + " did not override  calculateOutputDataTypes()! This function has not been implemented for " + getClass().getName());
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DifferentialFunction that = (DifferentialFunction) o;

        if (inPlace != that.inPlace) return false;
        if (scalarValue != null ? !scalarValue.equals(that.scalarValue) : that.scalarValue != null) return false;
        if (!Arrays.equals(dimensions, that.dimensions)) return false;
        return ownName != null ? ownName.equals(that.ownName) : that.ownName == null;
    }

    @Override
    public int hashCode() {
        int result = 31;
        result = 31 * result + (ownName != null ? ownName.hashCode() : 0);
        return result;
    }

    /**
     * The opName of this function in onnx
     * @return
     */
    public  String[] onnxNames() {
        return new String[] {onnxName()};
    }

    /**
     * The opName of this function tensorflow
     *
     * @return
     */
    public  String[] tensorflowNames() {
        return new String[] {tensorflowName()};
    }

    /**
     * The opName of this function in onnx
     * @return
     */
    public abstract String onnxName();

    /**
     * The opName of this function tensorflow
     *
     * @return
     */
    public abstract String tensorflowName();

    public int getNumOutputs(){return -1;}

    /**
     * Clear the input and output INDArrays, if any are set
     */
    public abstract void clearArrays();

    public boolean needsConfigure() {
        return false;
    }

}
