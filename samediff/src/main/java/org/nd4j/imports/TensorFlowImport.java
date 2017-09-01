package org.nd4j.imports;

import com.google.common.primitives.Ints;
import com.google.protobuf.TextFormat;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class provides TensorFlow graphs & models import
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class TensorFlowImport {

    /**
     *
     * @param graphFile
     * @return
     */
    public static SameDiff importGraph(File graphFile) {
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            GraphDef def = GraphDef.parseFrom(bis);
            return importGraph(def);
        } catch (Exception e) {
            try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis); BufferedReader reader = new BufferedReader(new InputStreamReader(bis))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line).append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                GraphDef def = builder.build();
                return importGraph(def);
            } catch (Exception e2) {
                e2.printStackTrace();
                throw new ND4JIllegalStateException("Can't parse graph: unknown format");
            }
        }
    }

    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    public static SameDiff importGraph(GraphDef tfGraph) {
        SDGraph graph = new SDGraph(true);

        SameDiff diff = SameDiff.builder()
                .graph(graph)
                .build();

        graph.setSameDiff(diff);

        Map<String, Integer> reverseVertexMap = new HashMap<>();

        int nodesCnt = 0;
        for (NodeDef tfNode :tfGraph.getNodeList()) {
            log.info("Node name: {}; Op: {};", tfNode.getName(), tfNode.getOp());


            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

            Map<String, AttrValue> attributes = tfNode.getAttrMap();





            if (isConst || isVar || isPlaceholder) {
                List<Integer> dimensions = new ArrayList<>();
                SDVariable variable = SDVariable.builder()
                        .varName(tfNode.getName())
                        .build();

                NDArrayInformation varInformation = NDArrayInformation.builder()
                        .id(tfNode.getName())
                        .build();

                Vertex<NDArrayInformation> vertex = new Vertex<>(++nodesCnt, varInformation);

                reverseVertexMap.put(tfNode.getName(), nodesCnt);

                int[] arrayShape = null;

                if (attributes.containsKey("dtype")) {
                    AttrValue dtype = attributes.get("dtype");

                    dtype.getList();
                }

                if (attributes.containsKey("shape")) {
                    AttrValue shape = attributes.get("shape");
                    int dims = shape.getShape().getDimCount();
                    if (dims > 0) {

                        // even vector is 2d in nd4j
                        if (dims == 1)
                            dimensions.add(1);

                        for (int e = 0; e < dims; e++) {
                            // TODO: eventually we want long shapes :(
                            dimensions.add((int) shape.getShape().getDim(e).getSize());
                        }
                    }

                    arrayShape = Ints.toArray(dimensions);

                    variable.setShape(arrayShape);
                }

                if (attributes.containsKey("value")) {
                    // value of?
                    AttrValue value = attributes.get("value");

                    //DataType type = value.

                    TensorProto tensor = value.getTensor();
                    log.info("Dtype: {}", tensor.getDtype());

                    INDArray array = getNDArrayFromTensor(tensor);
                    variable.setShape(array.shape());
                    variable.setArr(array);
                }

                diff.addVariable(variable);

                graph.addVertex(vertex);
            } else {
                // operation node
                NDArrayInformation varInformation = NDArrayInformation.builder()
                        .id(tfNode.getName())
                        .build();

                Vertex<NDArrayInformation> vertex = new Vertex<>(++nodesCnt, varInformation);
                graph.addVertex(vertex);

                OpState opState = getOpStateFromNodeDef(tfNode, tfNode.getInputCount());
                opState.setResult(varInformation);


                for (int e = 0; e < tfNode.getInputCount(); e++) {
                    String input = tfNode.getInput(e);

                    Integer id = reverseVertexMap.get(input);

                    if (id == null)
                        throw new ND4JIllegalStateException("Unknown input: [" + input + "]");

                    graph.addEdge(id, nodesCnt, opState, true);
                }
            }
        }



        return diff;
    }

    protected static INDArray getNDArrayFromTensor(TensorProto tfTensor) {
        int[] arrayShape = null;
        List<Integer> dimensions = new ArrayList<>();

        // building shape first
        int dims = tfTensor.getTensorShape().getDimCount();
        if (dims > 0) {
            // even vector is 2d in nd4j
            if (dims == 1)
                dimensions.add(1);

            for (int e = 0; e < dims; e++) {
                // TODO: eventually we want long shapes :(
                dimensions.add((int) tfTensor.getTensorShape().getDim(e).getSize());
            }
        }
        arrayShape = Ints.toArray(dimensions);

        if (tfTensor.getDtype() == DataType.DT_INT32) {
            // valueOf
            if (tfTensor.getIntValCount() == 1) {
                int val = tfTensor.getIntVal(0);

                INDArray array = Nd4j.valueArrayOf(arrayShape, (double) val);
                return array;
            } else {
                double[] jArray = new double[tfTensor.getIntValCount()];
                for (int e = 0; e < tfTensor.getIntValCount(); e++) {
                    jArray[e] = (double) tfTensor.getIntVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_FLOAT) {
            if (tfTensor.getFloatValCount() == 1) {
                float val = tfTensor.getFloatVal(0);

                INDArray array = Nd4j.valueArrayOf(arrayShape, (double) val);
                return array;
            } else {
                float[] jArray = new float[tfTensor.getFloatValCount()];
                for (int e = 0; e < tfTensor.getFloatValCount(); e++) {
                    jArray[e] = tfTensor.getFloatVal(e);
                }

                // FIXME: we're missing float[] signature
                INDArray array = Nd4j.create(ArrayUtil.toDoubles(jArray), arrayShape, 0, 'c');
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_DOUBLE) {
            if (tfTensor.getDoubleValCount() == 1) {
                double val = tfTensor.getDoubleVal(0);

                INDArray array = Nd4j.valueArrayOf(arrayShape, val);
                return array;
            } else {
                double[] jArray = new double[tfTensor.getDoubleValCount()];
                for (int e = 0; e < tfTensor.getDoubleValCount(); e++) {
                    jArray[e] =  tfTensor.getDoubleVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            }
        } else {
            throw new UnsupportedOperationException("Unknown dataType found: [" + tfTensor.getDtype() + "]");
        }
    }

    protected static OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs) {
        String lc = tfNode.getOp().toLowerCase();
        log.info("Looking for [{}] op...", lc);
        if (numInputs > 0 && numInputs <= 2) {
            int opNum = Nd4j.getOpFactory().getOpNumIfExists(lc);
            if (opNum >= 0) {
                OpState opState = OpState.builder()
                        .opType(OpState.opTypeFromOp(Nd4j.getOpFactory().getOpByName(lc)))
                        .opNum(opNum)
                        .opName(lc)
                        .build();

                return opState;
            }
        }

        OpState opState = OpState.builder()
                .opType(OpState.OpType.CUSTOM)
                .opNum(-1)
                .opName(tfNode.getOp())
                .build();

        return opState;
    }
}
