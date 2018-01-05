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
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

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
    protected Number scalarValue;


    @Getter
    @Setter
    @JsonIgnore
    protected int[] dimensions;

    @JsonIgnore
    protected Object[] extraArgs;

    //array initialized method being called
    @JsonIgnore
    protected boolean isArrayInit = false;

    //array already initialized
    @JsonIgnore
    protected  boolean arrayInitialized = false;

    @Getter
    @Setter
    private String ownName;

    public DifferentialFunction() {
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
        return Collections.emptyMap();
    }


    /**
     * Return function properties for the given function
     * @return
     */
    public FunctionProperties asProperties() {
        return FunctionProperties.builder()
                .name(opName())
                .fieldNames(propertiesForFunction())
                .build();
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

    public DifferentialFunction(SameDiff sameDiff, SDVariable[] args) {
        this(sameDiff,false,args);
    }

    public DifferentialFunction(SameDiff sameDiff, boolean inPlace, SDVariable[] args) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        setInstanceId();
        if(sameDiff != null)
            sameDiff.addArgsFor(args,this);
        for(int i = 0; i < args.length; i++) {
            if(args[i].isPlaceHolder()) {
                sameDiff.addPropertyToResolve(this,args[i].getVarName());
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
     * Return the output functions for this differential function.
     * @return
     */
    public abstract SDVariable[] outputVariables(String baseName);



    @JsonIgnore
    public  boolean isVariable() {
        return false;
    }








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
     * Returns true if this
     * function has place holder inputs
     * @return
     */
    public boolean hasPlaceHolderInputs() {
        val args = args();
        for(val arg : args)
            if(sameDiff.hasPlaceHolderVariables(arg().getVarName()))
                return true;
        return false;
    }

    @Override
    public abstract String toString();



    public boolean isConstant() {
        return false;
    }

    /**
     * Return the arguments for a given function
     * @return the arguments for a given function
     */
    public  SDVariable[] args() {
        return sameDiff.getInputVariablesForFunction(this);
    }




    public void resolvePropertiesFromSameDiffBeforeExecution() {

    }

    public SDVariable arg() {
        return args()[0];
    }


    public List<SDVariable> diff(List<SDVariable> i_v1) {
        List<SDVariable> vals = doDiff(i_v1);
        val outputVars = args();
        for(int i = 0; i < vals.size(); i++) {
            SDVariable var = outputVars[i];
            SDVariable grad = var.getGradient();
            if(grad != null) {
                SDVariable gradVar =  f().addi(grad, vals.get(i));

            }
            else {
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

    public String opName() {
        throw new UnsupportedOperationException();
    }


    public Op.Type opType() {
        throw new UnsupportedOperationException();
    }


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


    public void fillInArrays() {
        if(this instanceof Op) {
            Op op = (Op) this;
            op.setX(getX());
            //y is often optional for many problems
            if(args().length > 1)
                op.setY(getY());
            op.setZ(getZ());
        }
        else
            throw new IllegalStateException("Unable to fill in arrays. Type must be an operation.");
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
     * Calculate the output shape for this op
     * @return
     */
    public List<int[]> calculateOutputShape() {
        throw new UnsupportedOperationException();
    }




    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        DifferentialFunction that = (DifferentialFunction) o;

        if (inPlace != that.inPlace) return false;
        if (isArrayInit != that.isArrayInit) return false;
        if (arrayInitialized != that.arrayInitialized) return false;
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
        result = 31 * result + (isArrayInit ? 1 : 0);
        result = 31 * result + (arrayInitialized ? 1 : 0);
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


}
