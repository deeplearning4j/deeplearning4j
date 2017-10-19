package org.nd4j.linalg.api.ops.impl.controlflow;

import com.google.common.primitives.Ints;
import lombok.Builder;
import lombok.Getter;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.impl.transforms.Variable;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * Equivalent to tensorflow's while loop
 * Takes in:
 * loopVars
 * loop body
 * condition
 *
 * runs loop till condition is false.
 * @author Adam Gibson
 */
public class While extends DifferentialFunction implements CustomOp {

    @Getter
    private SameDiff.SameDiffFunctionBody loopBody;
    @Getter
    private SameDiff.SameDiffConditional conditional;
    @Getter
    private NDArrayVertex[] outputs;
    @Getter
    protected DifferentialFunction[] outputFunctions;

    private List<int[]> outputShapes;

    private List<INDArray> inputArrays;
    private List<INDArray> outputArrays;


    /**
     * Construct with relevant loop variables,
     * a loop body, and the proper conditionals.
     * @param loopVars the variables to use for the loop
     * @param loopBody the body of the loop
     * @param conditional the condition of termination for th eloop
     */
    @Builder
    public While(Variable[] loopVars, SameDiff.SameDiffFunctionBody loopBody, SameDiff.SameDiffConditional conditional) {
        this.loopBody = loopBody;
        this.conditional = conditional;
        this.args = loopVars;
        addEdges(loopVars[0].getSameDiff(),
                opName(),
                Op.Type.LOOP,
                null);
    }



    protected void addEdges(SameDiff sameDiff,
                            String opName,
                            Op.Type opType,
                            Object[] extraArgs) {
        for(DifferentialFunction input : args()) {
            validateFunctionReference(input);
            validateDifferentialFunctionGraph(input);
        }


        List<int[]> outputShapes = this.calculateOutputShape();
        this.outputShapes = outputShapes;
        int[] outputVertexIds = new int[outputShapes.size()];
        List<Integer> inputs = new ArrayList<>();
        for(int i = 0; i < args().length; i++) {
            DifferentialFunction differentialFunction = args()[i];
            List<DifferentialFunction> outputs = differentialFunction.outputs();
            for(DifferentialFunction output : outputs) {
                for(int vertexId : output.getOutputVertexIds()) {
                    if(!inputs.contains(vertexId))
                        inputs.add(vertexId);
                }
            }

        }

        this.outputs = new NDArrayVertex[outputShapes.size()];
        this.outputFunctions = new DifferentialFunction[outputShapes.size()];
        NDArrayInformation[] resultInfo = new NDArrayInformation[outputShapes.size()];
        for(int i = 0; i < outputShapes.size(); i++) {
            NDArrayInformation arrInfo = createOutputInfo(outputShapes.get(i),opName, UUID.randomUUID().toString());
            int nextVertexId = sameDiff.graph().nextVertexId();
            Variable variable = sameDiff.setupFunction(new Variable(sameDiff,opName + "-" +nextVertexId + "-" + i,arrInfo,nextVertexId));

            outputVertexIds[i] = variable.getVertex().vertexID();
            resultInfo[i] = arrInfo;
            this.outputs[i] = variable.getVertex();
            this.outputFunctions[i] = variable;
        }

        int[] inputIds = Ints.toArray(inputs);


        String[] vertexIds = sameDiff.generateVertexIds(Ints.concat(inputIds,outputVertexIds));
        OpState  opState = OpState.builder()
                .opType(opType).inPlace(inPlace)
                .differentialFunction(this)
                .opName(opName)
                .id(opName + "(" + vertexIds +  ")")
                .vertexIds(sameDiff.generateVertexIds(Ints.concat(inputIds,outputVertexIds)))
                .extraArgs(extraArgs)
                .results(resultInfo)
                .build();


        /**
         * Create 1 opstate with all of the vertex ids
         * with all inputs and outputs representing the edge.
         */
        sameDiff.graph().addEdge(
                inputIds,
                outputVertexIds,
                opState,true);




        this.opState = opState;




    }


    protected NDArrayInformation createOutputInfo(int[] shape,String id,String arrId) {
        return NDArrayInformation.builder()
                .arrId(arrId)
                .id(id)
                .shape(shape).build();
    }



    @Override
    public List<DifferentialFunction> doDiff(List<DifferentialFunction> f1) {
        return null;
    }

    @Override
    public String toString() {
        return opName();
    }

    @Override
    public String opName() {
        return "while";
    }

    @Override
    public long opHash() {
        return 0;
    }

    @Override
    public boolean isInplaceCall() {
        return false;
    }

    @Override
    public List<INDArray> getInputArguments() {
        if(inputArrays == null) {
            inputArrays = new ArrayList<>(args().length);
            for(int i  = 0; i < args().length; i++) {
                inputArrays.add(sameDiff.getArrayFor((Variable) args()[i]));
            }
        }

        return inputArrays;
    }

    @Override
    public List<INDArray> getOutputArguments() {
        if(outputArrays == null) {
            outputArrays = new ArrayList<>(outputFunctions.length);
            for(int i  = 0; i < outputFunctions.length; i++) {
                Variable outputArray = (Variable) sameDiff.getFunctionInstances().get(getOutputVertexIds()[i]);
                outputArrays.add(sameDiff.getArrayFor(outputArray));
            }
        }

        return outputArrays;
    }

    @Override
    public List<Integer> getIArguments() {
        return Collections.emptyList();
    }

    @Override
    public List<Double> getTArguments() {
        return Collections.emptyList();
    }

    @Override
    public List<int[]> calculateOutputShape() {
        List<int[]> ret =  new ArrayList<>();
        for(DifferentialFunction var : args()) {
            ret.add(var.getShape());
        }
        return ret;
    }
}
