package org.nd4j.autodiff.samediff;

import com.google.common.base.Preconditions;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ZeroInitScheme;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 *
 * A variable representing a component within a
 * {@@link SameDiff} graph.
 *
 * SDVariable is used for symbolic declaration
 * of equations.
 *
 * @author Adam Gibson
 *
 */
@Data
public class SDVariable extends DifferentialFunction implements Serializable {
    private INDArray arr;
    @Getter
    @Setter
    private String varName;
    private SDVariable gradient;
    private SDVariable forwardVariable;
    protected DifferentialFunction differentialFunction;
    @Getter
    @Setter
    protected WeightInitScheme weightInitScheme;
    @Builder
    private SDVariable(DifferentialFunction differentialFunction,
                       String varName,
                       INDArray arr,
                       OpState opState,
                       SameDiff sameDiff,
                       int[] shape,
                       WeightInitScheme weightInitScheme,
                       NDArrayVertex ndArrayVertex,
                       int[] vertexId) {
        this.shape =  shape;
        this.differentialFunction = differentialFunction;
        this.varName = varName;
        this.vertex = ndArrayVertex;
        this.weightInitScheme = weightInitScheme;
        this.arr = arr;
        this.vertexId = vertexId;
        if(shape == null) {
            if(arr != null) {
                this.shape = arr.shape();
            }
            else
                throw new ND4JIllegalStateException("Variable must have a shape.");
        }

        if(opState == null) {
            if(differentialFunction != null)
                this.opState = differentialFunction.getOpState();
            else
                this.opState = OpState.builder()
                        .opType(Op.Type.RETURN)
                        .inPlace(true)
                        .results(new SDVariable[] {this})
                        .vertexIds(ArrayUtil.convertToString(vertexId))
                        .opName(varName)
                        .build();
        }


        if(weightInitScheme == null) {
            this.weightInitScheme = new ZeroInitScheme('f');
        }

        this.sameDiff = sameDiff;
        if(differentialFunction != null) {
            this.vertexId = differentialFunction.getVertexId();
        }



    }

    @Override
    public boolean isVariable() {
        return true;
    }


    @Override
    public SDVariable getResult() {
        return this;
    }

    @Override
    public int[] getResultShape() {
        return getShape();
    }


    public void setArr(INDArray arr) {
        if(arr == null) {
            return;
        }

        this.arr = arr;
        if(differentialFunction instanceof Op) {
            Op op = (Op) differentialFunction;
            op.setZ(arr);
        }


    }

    /**
     * A getter for the allocated ndarray
     * with this {@link SDVariable}.
     *
     * This getter will lazy initialize an array if one is not found
     * based on the associated shape and {@link WeightInitScheme}
     * if neither are found, an {@link ND4JIllegalStateException}
     * is thrown.
     *
     * If a {@link DifferentialFunction} is defined, note that
     * its getArr() method is called instead.
     * @return the {@link INDArray} associated with this variable.
     */
    public INDArray getArr() {
        if(arr != null)
            return arr;

        //initialize value if it's actually a scalar constant (zero or 1 typically...)
        if(getScalarValue() != null && ArrayUtil.prod(getShape()) == 1) {
            INDArray arr = Nd4j.valueArrayOf(getShape(),
                    getScalarValue().doubleValue());
            sameDiff.associateArrayWithVariable(arr,this);
            setArr(arr);
        }
        else {
            INDArray newAlloc = getWeightInitScheme().create(getShape(),Nd4j.zeros(getShape(),getWeightInitScheme().order()));
            sameDiff.associateArrayWithVariable(newAlloc,this);
            setArr(newAlloc);

        }

        return arr;
    }

    public INDArray getArr(boolean requireArray) {
        if(arr == null && requireArray) {

            if(this.arr == null && sameDiff.getFunction("grad") != null) {
                this.arr = sameDiff.getFunction("grad").getVariable(varName).getArr(requireArray);
            }

            if(arr == null) {
                throw new IllegalStateException("Unable to get array. No vertex info or array field definition found.");
            }

        }

        return arr;
    }

    /**
     * Nicer looking alias
     * for the gradient variable.
     * The gradient variable is meant to be an
     * a variable representation
     * of the gradient represented
     * in the underlying {@link DifferentialFunction}
     * @return
     */
    public SDVariable gradient() {
        return getGradient();
    }

    /**
     * A getter for the variable gradient.
     * Note here that a lazy initialization of the
     * gradient variable will happen if the gradient
     * isn't present at this variable's initialization
     * but is set later.
     * @return
     */
    public SDVariable getGradient() {

        DifferentialFunction grad =  (this.getDifferentialFunction() != null ?  this.getDifferentialFunction().getGradient() : gradient);
        INDArray arr  = null;
        if(grad instanceof Op) {
            Op o = (Op) grad;
            arr = o.z();
        }
        else if(grad instanceof SDVariable) {
            SDVariable sdVariable = (SDVariable) grad;
            arr = sdVariable.getArr();
        }

        if(gradient == null && differentialFunction != null && differentialFunction.getGradient() != null) {
            this.gradient = differentialFunction != null && differentialFunction.getGradient() != null ? SDVariable.builder()
                    .sameDiff(sameDiff)
                    .differentialFunction(differentialFunction.getGradient())
                    .varName(varName + "-grad")
                    .arr(arr)
                    .shape(differentialFunction.getGradient() != null ? differentialFunction.getGradient().getResultShape() : null)
                    .build() : null;

            if(sameDiff.isDebugMode() && this.gradient != null) {
                sameDiff.addVariable(this.gradient);
            }
        }




        if(this.gradient != null) {
            this.gradient.setForwardVariable(this);
        }


        return gradient;
    }

    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        throw new ND4JIllegalStateException("Unable to differentiate a variable! Must be a function.");
    }

    /**
     *
     * @param gradient
     */
    public void setGradient(SDVariable gradient) {
        this.gradient = gradient;
        this.gradient.setForwardVariable(this);
    }





    /**
     * Returns the shape of this variable
     * @return
     */
    public int[] getShape() {
        if(shape != null)
            return shape;
        if(differentialFunction == null)
            throw new IllegalStateException("Unable to infer shape. Function is null.");
        OpState opState =  differentialFunction.getOpState();
        if(opState == null) {
            return differentialFunction.getResultShape();
        }

        return opState.getResults()[0].getShape();

    }


    /**
     *
     * @return
     */
    public boolean isAllocated() {
        return arr != null;
    }

    /**
     *
     */
    public void allocate() {
        if(arr == null)
            arr = Nd4j.zeros(getShape());
    }


    /**
     *
     * @return
     */
    public SDVariable dup() {
        return SDVariable.builder()
                .differentialFunction(differentialFunction)
                .varName(varName)
                .shape(shape)
                .sameDiff(sameDiff)
                .arr(arr != null ? arr.dup() : null)
                .build();
    }

    private int[] getTransformOutputShape(SDVariable other) {
        if(shape == null)
            return other.getShape();
        if(ArrayUtil.prod(shape) == 1) {
            return other.getShape();
        }

        return getShape();
    }





    //scalars

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(double sameDiffVariable) {
        return rsub(sameDiff.generateVariableName("rsub",false,this),sameDiffVariable);
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(double sameDiffVariable) {
        return rdiv(sameDiff.generateVariableName("rdiv",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(double sameDiffVariable) {
        return add(sameDiff.generateVariableName("add",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(double sameDiffVariable) {
        return sub(sameDiff.generateVariableName("sub",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(double sameDiffVariable) {
        return div(sameDiff.generateVariableName("div",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(double sameDiffVariable) {
        return mul(sameDiff.generateVariableName("mul",false,this),sameDiffVariable);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(double sameDiffVariable) {
        return rsubi(sameDiff.generateVariableName("rsubi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(double sameDiffVariable) {
        return rdivi(sameDiff.generateVariableName("rdivi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(double sameDiffVariable) {
        return addi(sameDiff.generateVariableName("addi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(double sameDiffVariable) {
        return subi(sameDiff.generateVariableName("subi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(double sameDiffVariable) {
        return divi(sameDiff.generateVariableName("divi",false,this),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(double sameDiffVariable) {
        return muli(sameDiff.generateVariableName("muli",false,this),sameDiffVariable);

    }



    //end scalars


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(SDVariable sameDiffVariable) {
        return rsub(sameDiff.generateVariableName("rsub",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(SDVariable sameDiffVariable) {
        return rdiv(sameDiff.generateVariableName("rdiv",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(SDVariable sameDiffVariable) {
        return add(sameDiff.generateVariableName("add",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(SDVariable sameDiffVariable) {
        return sub(sameDiff.generateVariableName("sub",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(SDVariable sameDiffVariable) {
        return div(sameDiff.generateVariableName("div",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(SDVariable sameDiffVariable) {
        return mul(sameDiff.generateVariableName("mul",false,this,sameDiffVariable),sameDiffVariable);

    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(SDVariable sameDiffVariable) {
        return rsubi(sameDiff.generateVariableName("rsubi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(SDVariable sameDiffVariable) {
        return rdivi(sameDiff.generateVariableName("rdivi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(SDVariable sameDiffVariable) {
        return addi(sameDiff.generateVariableName("addi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(SDVariable sameDiffVariable) {
        return subi(sameDiff.generateVariableName("subi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(SDVariable sameDiffVariable) {
        return divi(sameDiff.generateVariableName("divi",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(SDVariable sameDiffVariable) {
        return muli(sameDiff.generateVariableName("muli",false,this,sameDiffVariable),sameDiffVariable);

    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().rsub(getFunction(this),sameDiffVariable);

        SDVariable ret = SDVariable.builder()
                .varName(varName )
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().rdiv(getFunction(this),sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName )
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().add(getFunction(this),sameDiffVariable);
        SDVariable ret = SDVariable.builder()
                .varName(varName )
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(String varName, double sameDiffVariable) {
        DifferentialFunction right = getFunction(this);
        DifferentialFunction result = sameDiff.f().sub(right,sameDiffVariable);
        SDVariable ret =  SDVariable.builder()
                .varName(varName + " - " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(result)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().div(getFunction(this),sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName + " / " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().mul(getFunction(this)
                ,sameDiffVariable);
        SDVariable ret = SDVariable.builder()
                .varName(varName + " * " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().rsubi(getFunction(this),sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName + " - " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().rdivi(getFunction(this)
                ,sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName + " / " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().addi(getFunction(this),sameDiffVariable);

        SDVariable ret = SDVariable.builder()
                .varName(varName )
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().subi(getFunction(this),sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName + " - " + "scalar")
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().divi(getFunction(this),sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName)
                .arr(null)
                .sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(String varName, double sameDiffVariable) {
        DifferentialFunction function = sameDiff.f().muli(getFunction(this),sameDiffVariable);

        SDVariable ret =  SDVariable.builder().sameDiff(getSameDiff())
                .varName(varName)
                .arr(null).sameDiff(getSameDiff())
                .shape(getTransformOutputShape(this))
                .differentialFunction(function)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }



    //end scalars


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsub(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret =  SDVariable.builder().sameDiff(getSameDiff())
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().rsub(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdiv(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret = SDVariable.builder().sameDiff(getSameDiff())
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().rdiv(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable add(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);
        SDVariable ret =  SDVariable.builder()
                .sameDiff(getSameDiff())
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().add(getFunction(this),getFunction(sameDiffVariable)))
                .build();

        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable sub(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction left = getFunction(this);
        DifferentialFunction right = getFunction(sameDiffVariable);
        DifferentialFunction result = sameDiff.f().sub(left,right);
        SDVariable ret =  SDVariable.builder()
                .varName(varName)
                .arr(null)
                .sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable div(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().div(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable mul(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction left = getFunction(this);
        DifferentialFunction right = getFunction(sameDiffVariable);
        Preconditions.checkState(left != null,"Left input is null!");
        Preconditions.checkState(right != null,"Right input is null!");

        DifferentialFunction result = sameDiff.f().mul(left,right);

        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }


    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rsubi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret =  SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().rsubi(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable rdivi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().rdivi(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable addi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().addi(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable subi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction left = getFunction(this);
        DifferentialFunction right = getFunction(sameDiffVariable);
        DifferentialFunction result = sameDiff.f().subi(left,right);
        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null)
                .sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable divi(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff()).shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(sameDiff.f().divi(getFunction(this),getFunction(sameDiffVariable)))
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }

    /**
     *
     * @param sameDiffVariable
     * @return
     */
    public SDVariable muli(String varName, SDVariable sameDiffVariable) {
        assertShapeEquals(sameDiffVariable);

        DifferentialFunction left = getFunction(this);
        DifferentialFunction right = getFunction(sameDiffVariable);
        DifferentialFunction result = sameDiff.f().muli(left,right);
        SDVariable ret = SDVariable.builder()
                .varName(varName)
                .arr(null).sameDiff(sameDiffVariable.getSameDiff())
                .shape(getTransformOutputShape(sameDiffVariable))
                .differentialFunction(result)
                .build();
        sameDiff.addVariable(ret);
        return ret;
    }



    /**
     * Evaluate the result of this variable
     * @return
     */
    public INDArray eval() {
        SameDiff exec = sameDiff.dup();
        exec.defineFunction("output", new SameDiff.SameDiffFunctionDefinition() {
            @Override
            public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
                return new SDVariable[] { SDVariable.this};
            }
        });

        SDVariable output = exec.invokeFunctionOn("output",exec);
        return output.getSameDiff().execAndEndResult();
    }





    private void assertShapeEquals(SDVariable variable) {
        if(!Arrays.equals(shape,variable.getShape()) && ArrayUtil.prod(variable.getShape()) != 1 && Shape.broadcastOutputShape(shape,variable.shape) == null) {
            throw new IllegalArgumentException("Input shape must be the same as this shape " + Arrays.toString(shape) + " and shape was " + Arrays.toString(variable.getShape()));
        }
    }


    /**
     * Return the underlying differential
     * function
     * or array field.
     * @param variable
     * @return
     */
    public static DifferentialFunction getFunction(SDVariable variable) {
        if(variable == null)
            throw new IllegalArgumentException("Unable to get function for null variable");
        return variable.getDifferentialFunction() != null ? variable.getDifferentialFunction() : variable;
    }

    @Override
    public String toString() {
        return varName;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        SDVariable variable = (SDVariable) o;

        if (arr != null ? !arr.equals(variable.arr) : variable.arr != null) return false;
        if (varName != null ? !varName.equals(variable.varName) : variable.varName != null) return false;
        return differentialFunction != null ? differentialFunction.equals(variable.differentialFunction) : variable.differentialFunction == null;
    }

    @Override
    public int hashCode() {
        int result = 0;
        result = 31 * result + (arr != null ? arr.hashCode() : 0);
        result = 31 * result + (varName != null ? varName.hashCode() : 0);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + (differentialFunction != null ? differentialFunction.hashCode() : 0);
        return result;
    }
}
