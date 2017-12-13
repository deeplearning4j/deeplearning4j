package org.nd4j.autodiff.functions;

import com.rits.cloning.Cloner;
import lombok.*;
import lombok.extern.slf4j.Slf4j;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.UUID;


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
    protected String instanceId;

    public DifferentialFunction() {
        this.instanceId = UUID.randomUUID().toString();
    }

    /**
     * Initialize the function from the given
     * {@link NodeDef}
     * @param nodeDef
     */
    public DifferentialFunction(SameDiff sameDiff,NodeDef nodeDef, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        this.sameDiff = sameDiff;
        this.instanceId = UUID.randomUUID().toString();
        initFromTensorFlow(nodeDef, sameDiff,attributesForNode ,graph);
    }

    /**
     * Iniitialize the function from the given
     * {@link onnx.OnnxProto3.NodeProto}
     * @param node
     */
    public DifferentialFunction(SameDiff sameDiff,onnx.OnnxProto3.NodeProto node,Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        this.sameDiff = sameDiff;
        this.instanceId = UUID.randomUUID().toString();

        initFromOnnx(node, sameDiff, attributesForNode, graph);
    }


    /**
     *
     * @param sameDiff
     * @param extraArgs
     */
    public DifferentialFunction(SameDiff sameDiff,boolean inPlace, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        this.instanceId = UUID.randomUUID().toString();
        this.extraArgs = extraArgs;


    }


    /**
     *
     * @param sameDiff
     * @param extraArgs
     */
    public DifferentialFunction(SameDiff sameDiff, Object[] extraArgs) {
        this.sameDiff = sameDiff;
        this.instanceId = UUID.randomUUID().toString();
        this.extraArgs = extraArgs;

    }

    public DifferentialFunction(SameDiff sameDiff, SDVariable[] args) {
        this(sameDiff,false,args);
    }

    public DifferentialFunction(SameDiff sameDiff, boolean inPlace, SDVariable[] args) {
        this.sameDiff = sameDiff;
        this.inPlace = inPlace;
        this.instanceId = UUID.randomUUID().toString();
        val nodeIds = new int[args.length];
        for(int i = 0; i < args.length; i++) {
            nodeIds[i] = args[i].getVertexId();
        }

        if(sameDiff != null) {
            sameDiff.addArgsFor(nodeIds, this);
        }
    }





    /**
     * Return the output functions for this differential function.
     * @return
     */
    public abstract SDVariable[] outputVariables();





    @JsonIgnore
    public  boolean isVariable() {
        return false;
    }


    /**
     * Get the input vertex ids
     * for this function
     * @return
     */
    public int[] inputVertexIds() {
        val args = args();
        int[] ret = new int[args.length];
        for(int i = 0; i < args.length; i++) {
            ret[i] = args[i].getVertexId();
        }

        return ret;
    }


    /**
     * Get the input vertex ids
     * for this function
     * @return
     */
    public int[] outputVertexIds() {
        val args = outputVariables();
        int[] ret = new int[args.length];
        for(int i = 0; i < args.length; i++) {
            ret[i] = args[i].getVertexId();
        }

        return ret;
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



    //by default no op, used for certain situations like
    //place holder arrays
    public void initOutputWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        if(isArrayInit() || isArrayInitialized()) {
            return;
        }

        val shapeCalc = calculateOutputShape();
        if(hasPlaceHolderInputs() && shapeCalc != null && !shapeCalc.isEmpty()) {
            //update place holder shapes in case the shapes
            // need to be resolved
            //post adding the variables to the graph.
            if(sameDiff.shapeAlreadyExistsForVertexId(args()[0].getVertexId()))
                sameDiff.updateShapeForVertexId(args()[0].getVertexId(),shapeCalc.get(0));
            else
                sameDiff.putShapeForVertexId(args()[0].getVertexId(),shapeCalc.get(0));

        }

        this.arrayInitialized = true;
    }

    //by default no op, used for certain situations like
    //place holder arrays
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        if(isArrayInit() || isArrayInitialized()) {
            return;
        }

        for(val arg : args()) {
            arg.initWithArrays(arrayMap,extraArgs);
        }

    }



    /**
     * Returns true if this
     * function has place holder inputs
     * @return
     */
    public boolean hasPlaceHolderInputs() {
        val args = args();
        for(val arg : args)
            if(sameDiff.hasPlaceHolderVariables(arg.getVertexId()))
                return true;
        return false;
    }

    @Override
    public abstract String toString();



    public boolean isConstant() {
        return false;
    }

    public  SDVariable[] args() {
        return sameDiff.getInputVariablesForFunction(this);
    }

    public SDVariable arg() {
        return args()[0];
    }


    public List<SDVariable> diff(List<SDVariable> i_v1) {
        List<SDVariable> vals = doDiff(i_v1);
        val outputVars = args();
        for(int i = 0; i < outputVars.length; i++) {
            SDVariable var = outputVars[i];
            SDVariable grad = var.getGradient();
            if(grad != null) {
                SDVariable ret = f().addi(var, grad);
                sameDiff.updateVariableName(ret.getVertexId(),var.getVarName() + "-grad");
                sameDiff.setGradientForVertexId(var.getVertexId(),sameDiff.getVariableForVertexId(ret.getVertexId()));
                sameDiff.setForwardVariableForVertexId(ret.getVertexId(),var);
            }
            else {
                SDVariable gradVar = vals.get(i);
                sameDiff.setGradientForVertexId(var.getVertexId(), gradVar);
                sameDiff.setForwardVariableForVertexId(gradVar.getVertexId(),var);
            }
        }

        return vals;
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
        INDArray ret =  sameDiff.getArrForVertexId(args()[0].getVertexId());
        return ret;
    }

    @JsonIgnore
    private INDArray getY() {
        if(args().length > 1) {
            INDArray ret =  sameDiff.getArrForVertexId(args()[1].getVertexId());
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
        Cloner cloner = new Cloner();
        return cloner.deepClone(this);
    }





    /**
     * Calculate the output shape for this op
     * @return
     */
    public List<int[]> calculateOutputShape() {
        throw new UnsupportedOperationException();
    }


    public int maxDepthForArgs() {
        int depth = -1;
        for(SDVariable arg : args()) {
            if(arg == this)
                continue;
            depth = Math.max(arg.depth(),depth);
        }

        return depth;
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
        return instanceId != null ? instanceId.equals(that.instanceId) : that.instanceId == null;
    }

    @Override
    public int hashCode() {
        int result = 31;
        result = 31 * result + (inPlace ? 1 : 0);
        result = 31 * result + (scalarValue != null ? scalarValue.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(dimensions);
        result = 31 * result + (isArrayInit ? 1 : 0);
        result = 31 * result + (arrayInitialized ? 1 : 0);
        result = 31 * result + (instanceId != null ? instanceId.hashCode() : 0);
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

    protected int fromBoolean(boolean bool) {
        return bool ? 1 : 0;
    }



}
