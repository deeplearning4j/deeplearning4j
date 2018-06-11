package org.nd4j.autodiff.execution;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.execution.conf.ExecutionMode;
import org.nd4j.autodiff.execution.conf.ExecutorConfiguration;
import org.nd4j.autodiff.execution.conf.OutputMode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.FlatArray;
import org.nd4j.graph.FlatResult;
import org.nd4j.graph.FlatVariable;
import org.nd4j.graph.OpType;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class NativeGraphExecutioner implements GraphExecutioner {
    /**
     * This method returns Type of this executioner
     *
     * @return
     */
    @Override
    public Type getExecutionerType() {
        return Type.LOCAL;
    }


    /**
     * This method executes given graph and returns results
     *
     * PLEASE NOTE: Default configuration is used
     *
     * @param sd
     * @return
     */
    @Override
    public INDArray[] executeGraph(SameDiff sd) {
        return executeGraph(sd, ExecutorConfiguration.builder().outputMode(OutputMode.IMPLICIT).executionMode(ExecutionMode.SEQUENTIAL).profilingMode(OpExecutioner.ProfilingMode.DISABLED).build());
    }

    @Override
    public INDArray[] reuseGraph(SameDiff graph, Map<Integer, INDArray> inputs) {
        throw new UnsupportedOperationException();
    }

    public ByteBuffer convertToFlatBuffers(SameDiff sd, ExecutorConfiguration configuration, Map<Integer, Node> intermediate) {
        log.info("Configuration: {}", configuration);

        return sd.asFlatBuffers(configuration);
    }

    @Override
    public ByteBuffer convertToFlatBuffers(SameDiff sd, ExecutorConfiguration configuration) {
        return convertToFlatBuffers(sd, configuration, new HashMap<Integer, Node>());
    }

    /**
     * This method executes given graph and returns results
     *
     * @param sd
     * @return
     */
    @Override
    public INDArray[] executeGraph(SameDiff sd, ExecutorConfiguration configuration) {

        Map<Integer, Node> intermediate = new HashMap<>();

        ByteBuffer buffer = convertToFlatBuffers(sd, configuration, intermediate);

        BytePointer bPtr = new BytePointer(buffer);

        log.info("Buffer length: {}", buffer.limit());

        Pointer res  = NativeOpsHolder.getInstance().getDeviceNativeOps().executeFlatGraphFloat(null, bPtr);
        if (res == null)
            throw new ND4JIllegalStateException("Graph execution failed");

        // FIXME: this is BAD
        PagedPointer pagedPointer = new PagedPointer(res,1024 * 1024L);
        FlatResult fr = FlatResult.getRootAsFlatResult(pagedPointer.asBytePointer().asByteBuffer());


        log.info("VarMap: {}", sd.variableMap());

        INDArray[] results = new INDArray[fr.variablesLength()];

        for (int e = 0; e < fr.variablesLength(); e++) {
            FlatVariable var = fr.variables(e);
            log.info("Var received: id: [{}:{}/<{}>];", var.id().first(), var.id().second(), var.name());
            FlatArray ndarray = var.ndarray();


            INDArray val = Nd4j.createFromFlatArray(ndarray);
            results[e] = val;

            if (var.name() != null && sd.variableMap().containsKey(var.name())) {
                //log.info("VarName: {}; Exists: {}; NDArrayInfo: {};", var.opName(), sd.variableMap().containsKey(var.opName()), sd.getVertexToArray().containsKey(var.opName()));
  //              log.info("storing: {}; array: {}", var.name(), val);
                sd.associateArrayWithVariable(val, sd.variableMap().get(var.name()));

            } else {
                //log.info("Original id: {}; out: {}; out2: {}", original, sd.getVertexIdxToInfo().get(original), graph.getVariableForVertex(original));
                if (sd.variableMap().get(var.name()) != null) {
                    sd.associateArrayWithVariable(val,sd.getVariable(var.name()));
                } else {
                    log.warn("Unknown variable received: [{}]", var.name());

//                    throw new ND4JIllegalStateException("Unknown variable received as result: ["+ var.name() +"]");
                }
            }
        }


        return results;
    }



    public static long getOpNum(String name, Op.Type type) {
        if (type == Op.Type.CUSTOM)
            return Nd4j.getExecutioner().getCustomOperations().get(name.toLowerCase()).getHash();
        else
            return (long) Nd4j.getOpFactory().getOpNumByName(name);
    }

    public static byte getFlatOpType(Op.Type type) {
        switch (type) {
            case SCALAR:
                return OpType.SCALAR;
            case BROADCAST:
                return OpType.BROADCAST;
            case TRANSFORM:
                return OpType.TRANSFORM;
            case REDUCE:
                return OpType.ACCUMULATION;
            case INDEXREDUCE:
                return OpType.INDEX_ACCUMULATION;
            case CUSTOM:
                return OpType.CUSTOM;
            default:
                throw new UnsupportedOperationException("Unknown op type passed in: " + type);
        }
    }

    /**
     * This method executes
     *
     * @param id
     * @param variables
     * @return
     */
    @Override
    public INDArray[] executeGraph(int id, SDVariable... variables) {
        return new INDArray[0];
    }

    /**
     * This method stores given graph for future execution
     *
     * @param graph
     * @return
     */
    @Override
    public int registerGraph(SameDiff graph) {
        return 0;
    }


    @Override
    public INDArray[] importProto(File file) {
        // TODO: to be implemented
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
