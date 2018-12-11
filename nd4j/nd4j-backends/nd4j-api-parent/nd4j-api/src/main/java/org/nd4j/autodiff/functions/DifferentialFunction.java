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

package org.nd4j.autodiff.functions;

import com.rits.cloning.Cloner;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.descriptors.properties.AttributeAdapter;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.lang.reflect.Field;
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
    protected int[] dimensions;

    @JsonIgnore
    protected Object[] extraArgs;


    @Getter
    @Setter
    @JsonIgnore
    private String ownName;

    public DifferentialFunction() {
        this(true);
    }

    public DifferentialFunction(boolean sameDiff){
        //Only need instance ID if using function in context of SameDiff, not standard ND4J with INDArray args
        if(sameDiff)
            setInstanceId();
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
    }

    /**
     * Initialize the function from the given
     * {@link onnx.OnnxProto3.NodeProto}
     * @param node
     */
    public DifferentialFunction(SameDiff sameDiff,onnx.OnnxProto3.NodeProto node,Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        this.sameDiff = sameDiff;
        setInstanceId();
        initFromOnnx(node, sameDiff, attributesForNode, graph);
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

        for(val entry : fields.entrySet()) {
            try {
                ret.put(entry.getKey(),fields.get(entry.getKey()).get(this));
            } catch (IllegalAccessException e) {
                throw new RuntimeException("Unable to get property for field: " + entry.getKey(), e);
            }
        }

        return ret;
    }

    public void setPropertiesForFunction(Map<String,Object> properties){
        Map<String,Field> fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);
        for(String s : properties.keySet()){
            Field f = fields.get(s);
            if(f == null){
                log.warn("No fields found for property name {} for class {}", s, this.getClass().getName());
                continue;
            }
            setValueFor(f, properties.get(s));
        }
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
            e.printStackTrace();
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
    public void setValueFor(Field target, Object value) {
        if(value == null && target.getType().isPrimitive()) {
            throw new ND4JIllegalStateException("Unable to set primitive field " + target + " of type " + target.getClass()
                    + " using null value!");
        }

        if(value != null) {
            value = ensureProperType(target, value);
        }

        if(isConfigProperties()){
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
            while(f == null && currClass.getSuperclass() != null){
                currClass = currClass.getSuperclass();
                try{
                    f = currClass.getDeclaredField(propertyName);
                } catch (NoSuchFieldException e){
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
                target.set(this,value);
            } catch (IllegalAccessException e) {
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
        if(sameDiff != null) {
            sameDiff.addArgsFor(args, this);
            for (int i = 0; i < args.length; i++) {
                if (args[i].isPlaceHolder()) {
                    sameDiff.addPropertyToResolve(this, args[i].getVarName());
                }
            }
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
    public SDVariable outputVariable(){
        return outputVariables()[0];
    }


    public String[] outputVariablesNames(){
        SDVariable[] outputVars = outputVariables();
        String[] out = new String[outputVars.length];
        for( int i=0; i<out.length; i++ ){
            out[i] = outputVars[i].getVarName();
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
     * Shortcut for the {@link DifferentialFunctionFactory}
     * @return
     */
    public DifferentialFunctionFactory f() {
        return sameDiff.f();
    }


    /**
     * Return the arguments for a given function
     * @return the arguments for a given function
     */
    public  SDVariable[] args() {
        return sameDiff.getInputVariablesForFunction(this);
    }

    /**
     * Return the specified argument for this function
     * @param num Number of the argument. Must be in range 0 to numArgs - 1 inclusive
     * @return Specified argument
     */
    public SDVariable arg(int num){
        SDVariable[] args = args();
        Preconditions.checkNotNull(args, "Arguments are null for function %s", this.getOwnName());
        Preconditions.checkArgument(num >= 0 && num < args.length, "Invalid index: must be 0 to numArgs (0 <= idx < %s)", args.length);
        return args[num];
    }

    public String[] argNames(){
        SDVariable[] args = args();
        String[] out = new String[args.length];
        for( int i=0; i<args.length; i++ ){
            out[i] = args[i].getVarName();
        }
        return out;
    }


    /**
     * Resolve properties and arguments right before execution of
     * this operation.
     */
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        val properties = sameDiff.propertiesToResolveForFunction(this);
        val fields = DifferentialFunctionClassHolder.getInstance().getFieldsForFunction(this);
        val currentFields = this.propertiesForFunction();

        for(val property : properties) {
            //property maybe a variable which is only an array
            //just skip  if this is the case
            if(!fields.containsKey(property))
                continue;

            val var = sameDiff.getVarNameForFieldAndFunction(this,property);
            val fieldType = fields.get(property);
            val varArr = sameDiff.getArrForVarName(var);
            //already defined
            if(currentFields.containsKey(property)) {
                continue;
            }

            /**
             * Possible cause:
             * Might be related to output name alignment.
             *
             */
            if(varArr == null) {
                throw new ND4JIllegalStateException("Unable to set null array!");
            }

            if(fieldType.getType().equals(int[].class)) {
                setValueFor(fieldType,varArr.data().asInt());
            }

            else if(fieldType.equals(double[].class)) {
                setValueFor(fieldType,varArr.data().asDouble());
            }

            else if(fieldType.equals(int.class)) {
                setValueFor(fieldType,varArr.getInt(0));
            }

            else if(fieldType.equals(double.class)) {
                setValueFor(fieldType,varArr.getDouble(0));
            }

        }

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
        if(vals == null){
            throw new IllegalStateException("Error executing diff operation: doDiff returned null for op: " + this.opName());
        }

        val outputVars = args();
        boolean copied = false;
        for(int i = 0; i < vals.size(); i++) {
            SDVariable var = outputVars[i];
            SDVariable grad = var.getGradient();
            if(grad != null) {
                if(!copied){
                    //Don't mutate the original - this could mess with the original op's state!
                    vals = new ArrayList<>(vals);
                    copied = true;
                }

                SDVariable gradVar =  f().add(grad, vals.get(i));
                vals.set(i, gradVar);
                sameDiff.setGradientForVariableName(var.getVarName(), gradVar);
            } else {
                SDVariable gradVar = vals.get(i);
                sameDiff.updateVariableNameAndReference(gradVar,var.getVarName() + "-grad");
                sameDiff.setGradientForVariableName(var.getVarName(), gradVar);
                sameDiff.setForwardVariableForVarName(gradVar.getVarName(),var);

            }
        }

        return vals;
    }


    protected void setInstanceId() {
        if(ownName == null) {
            if(sameDiff == null)
                this.ownName = UUID.randomUUID().toString();
            else {
                int argIndex = 0;
                String varName = sameDiff.generateNewVarName(opName(),argIndex);
                while(sameDiff.functionExists(varName)) {
                    varName = sameDiff.generateNewVarName(opName(), argIndex);
                    argIndex++;
                }

                this.ownName = varName;
            }

            if(sameDiff != null && !(this instanceof SDVariable))
                sameDiff.putFunctionForId(ownName,this);
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
    private INDArray getX() {
        INDArray ret =  sameDiff.getArrForVarName(args()[0].getVarName());
        return ret;
    }

    @JsonIgnore
    private INDArray getY() {
        if(args().length > 1) {
            INDArray ret =  sameDiff.getArrForVarName(args()[1].getVarName());
            return ret;
        }
        return null;
    }

    @JsonIgnore
    private INDArray getZ() {
        if(isInPlace())
            return getX();
        SDVariable opId = outputVariables()[0];
        INDArray ret = opId.getArr();
        return ret;
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
     * {@link onnx.OnnxProto3.NodeProto}
     * @param node
     * @param initWith
     * @param attributesForNode
     * @param graph
     */
    public abstract void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph);



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
    public  DifferentialFunction dup() {
        Cloner cloner = SameDiff.newCloner();
        return cloner.deepClone(this);
    }





    /**
     * Calculate
     * the output shape for this op
     * @return
     */
    public List<LongShapeDescriptor> calculateOutputShape() {
        throw new ND4JIllegalStateException("calculateOutputShape() method leaked out for [" + this.opName() + "]");
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
        result = 31 * result + (inPlace ? 1 : 0);
        result = 31 * result + (scalarValue != null ? scalarValue.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(dimensions);
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

}
