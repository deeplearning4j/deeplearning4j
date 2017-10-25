package org.nd4j.imports;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Ints;
import com.google.protobuf.TextFormat;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.autodiff.graph.api.Edge;
import org.nd4j.autodiff.graph.api.Vertex;
import org.nd4j.autodiff.opstate.NDArrayInformation;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.impl.SDVariable;
import org.nd4j.graph.intermediate.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.*;

import java.io.*;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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
        GraphDef def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = GraphDef.parseFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = builder.build();
            } catch (Exception e2) {
                e2.printStackTrace();
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format: " + graphFile.getAbsolutePath());


        return importGraph(def);
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
                .vertexToArray(Maps.<String, INDArray>newHashMap())
                .variableMap(Maps.<String, SDVariable>newHashMap())
                .vertexIdxToInfo(Maps.<int[], NDArrayInformation>newHashMap())
                .build();

        graph.setSameDiff(diff);

        Map<String, Integer> reverseVertexMap = new HashMap<>();

        int nodesCnt = 0;
        val skipPoint = new AtomicLong(0);
        Set<String> skipList = new HashSet<>();
        val tfNodesList = tfGraph.getNodeList();
        for (NodeDef tfNode : tfNodesList) {
            log.debug("Node name: {}; Op: {};", tfNode.getName(), tfNode.getOp());

            if (tfNode.getOp().equalsIgnoreCase("enter")) {
                continue;
            }

            // if we've used forward-scan (i.e. for loops ) we can already have this node mapped
            if (skipList.contains(tfNode.getName()))
                continue;

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

            Map<String, AttrValue> attributes = tfNode.getAttrMap();



            if (isConst || isVar || isPlaceholder) {
                List<Integer> dimensions = new ArrayList<>();
                SDVariable variable = SDVariable.builder()
                        .sameDiff(diff)
                        .varName(tfNode.getName())
                        .build();

                NDArrayInformation varInformation = NDArrayInformation.builder()
                        .id(tfNode.getName())
                        .arrId(tfNode.getName())
                        .build();

                NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt,0, varInformation);

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

                    //DataType opType = value.

                    TensorProto tensor = value.getTensor();
                    log.debug("Dtype: {}", tensor.getDtype());

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

                NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt, 0,varInformation);
                graph.addVertex(vertex);

                OpState opState = getOpStateFromNodeDef(tfNode, tfNode.getInputCount());
                opState.setResults(new NDArrayInformation[]{varInformation});

                reverseVertexMap.put(tfNode.getName(), nodesCnt);


                for (int e = 0; e < tfNode.getInputCount(); e++) {
                    String input = tfNode.getInput(e);

                    Integer id = reverseVertexMap.get(input);

                    if (id == null)
                        throw new ND4JIllegalStateException("Unknown input: [" + input + "]");

                    graph.addEdge(new int[]{id}, new int[]{nodesCnt}, opState, true);
                }
            }
        }
        return diff;
    }

    /**
     * This method returns intermediate representation from TF protobuf file
     *
     * @param graphFile
     * @return
     */
    public static TGraph importIntermediate(File graphFile) {
        GraphDef def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = GraphDef.parseFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                GraphDef.Builder builder = GraphDef.newBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = builder.build();
            } catch (Exception e2) {
                //
            }
        }

        if (def == null)
            throw new ND4JIllegalStateException("Unknown format");


        return importIntermediate(def);
    }

    protected static TNode importWhileLoop(TGraph intermediateGraph, @NonNull Map<String, TIndex> reverseVertexMap, int startPosition, Set<String> skipList, List<NodeDef> nodes, @NonNull AtomicInteger varsCnt, @NonNull AtomicInteger nodesCnt) {


        val scopeCondition = new TScope(nodesCnt.incrementAndGet(), "scopeCondition");
        val scopeLoop = new TScope(nodesCnt.incrementAndGet(), "scopeLoop");

        intermediateGraph.addScope(scopeCondition);
        intermediateGraph.addScope(scopeLoop);

        val tNode = TNode.builder().id(nodesCnt.incrementAndGet())
                .inputs(TIndex.indices(TIndex.makeOf(scopeCondition.getId()), TIndex.makeOf(scopeLoop.getId())))
                .opName("while")
                .opNum(0)
                .opState(OpState.builder().opName("while").opNum(0).opType(Op.Type.LOOP).build())
                .build();

        log.info("Adding 2 new scopes for WHILE {}", tNode.getId());


        /**
         * Plan is simple:
         * 1) we read all declarations of variables used within loop
         * 2) we set up conditional scope
         * 3) we set up body scope
         * 4) ???
         * 5) PROFIT!
         */

        Map<String, TIndex> localReverseMap = new HashMap<>();

        // parsing declarations first. they all come as Enter ops
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("enter")) {
                skipList.add(tfNode.getName().toLowerCase());
                break;
            }

            skipList.add(tfNode.getName().toLowerCase());

            for (int e = 0; e < tfNode.getInputCount(); e++) {
                val input = tfNode.getInput(e);
                val idx = reverseVertexMap.get(input);
                log.info("Mapping [{}] to [{}]", input, idx);
            }
        }

        // now we're skipping Merge step, since we've already captured variables at Enter step
        int mergedCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("merge")) {
                skipList.add(tfNode.getName().toLowerCase());
                break;
            }

            skipList.add(tfNode.getName().toLowerCase());


            //localReverseMap.put(tfNode.getName().toLowerCase(), TIndex.makeOf(tNode.getId(), mergedCnt));
            reverseVertexMap.put(tfNode.getName(), TIndex.makeOf(tNode.getId(), mergedCnt++));
        }


        // now, we're adding conditional scope
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            // we're parsing up to condition
            if (tfNode.getOp().equalsIgnoreCase("LoopCond")) {
                skipList.add(tfNode.getName().toLowerCase());
                startPosition++;
                break;
            }

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = importVariable(tfNode, reverseVertexMap, varsCnt.decrementAndGet());
                log.info("Adding condition var [{}:{}]", var.getName(), var.getId());

                intermediateGraph.getVariableSpace().addVariable(var.getId(), var);
            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());

                val scopedNode = importNode(intermediateGraph, tfNode, reverseVertexMap, nodesCnt.incrementAndGet());
                scopedNode.setScoped(true);
                scopedNode.setScopeId(scopeCondition.getId());
                scopedNode.setScopeName(scopeCondition.getName());

                scopeCondition.addNode(scopedNode);
            }

            skipList.add(tfNode.getName().toLowerCase());
        }



        // time to skip some Switch calls
        int switchCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            // we're parsing up to condition
            if (!tfNode.getOp().equalsIgnoreCase("Switch"))
                break;

            reverseVertexMap.put(tfNode.getName(), TIndex.makeOf(tNode.getId(), switchCnt++));
            skipList.add(tfNode.getName().toLowerCase());
        }

        // now we're parsing Identity step
        int identityCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);


            if (!tfNode.getOp().equalsIgnoreCase("Identity")) {
                break;
            }

            val scopedNode = importNode(intermediateGraph, tfNode, reverseVertexMap, nodesCnt.incrementAndGet());
            scopedNode.setScopeId(scopeLoop.getId());
            scopedNode.setScopeName(scopeLoop.getName());

            // we overwrite inputs here, because that's always mapping to the While scope operands
            scopedNode.setInputs(Lists.newArrayList(TIndex.makeOf(tNode.getId(), identityCnt++)));

            scopeLoop.addNode(scopedNode);

            skipList.add(tfNode.getName().toLowerCase());
        }


        // parsing body scope
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (tfNode.getOp().equalsIgnoreCase("NextIteration")) {
                break;
            }


            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = importVariable(tfNode, reverseVertexMap, varsCnt.decrementAndGet());
                log.info("Adding body var [{}:{}]", var.getName(), var.getId());

                intermediateGraph.getVariableSpace().addVariable(var.getId(), var);
            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());

                val scopedNode = importNode(intermediateGraph, tfNode, reverseVertexMap, nodesCnt.incrementAndGet());
                scopedNode.setScoped(true);
                scopedNode.setScopeId(scopeLoop.getId());
                scopedNode.setScopeName(scopeLoop.getName());

                scopeLoop.addNode(scopedNode);
            }

            skipList.add(tfNode.getName().toLowerCase());
        }


        // skipping NextIterations, we just know when the Scope ends
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("NextIteration"))
                break;

            skipList.add(tfNode.getName().toLowerCase());
        }

        // we should also map While/Exit to libnd4j while
        int exitCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("Exit"))
                break;

            skipList.add(tfNode.getName().toLowerCase());

            reverseVertexMap.put(tfNode.getName(), TIndex.makeOf(tNode.getId(), exitCnt++));
        }


        log.info("-------------------------------------------");

        return tNode;
    }



    protected static TNode importNode(@NonNull TGraph intermediateGraph, @NonNull NodeDef tfNode, @NonNull Map<String, TIndex> reverseVertexMap, int nodeId) {
        reverseVertexMap.put(tfNode.getName(), TIndex.makeOf(nodeId, 0));
        val tNode = TNode.builder()
                .name(tfNode.getName())
                .id(nodeId)
                .opName(tfNode.getOp())
                .build();


        for (int e = 0; e < tfNode.getInputCount(); e++) {
            val input = tfNode.getInput(e);


            // input taken from mult
            if (input.startsWith("^")) {
                log.debug("Wow");
            } else if (input.contains(":")) {
                val split = input.split(":");

                if (split.length == 1) {
                    Integer id = reverseVertexMap.get(split[0]).getNode();

                    tNode.addInput(id);
                } else if (split.length == 2) {
                    Integer node = reverseVertexMap.get(split[0]).getNode();
                    Integer idx = Integer.valueOf(split[1]);

                    if (node == null) {
                        log.error("Can't find mapped node [{}]", input);
                        throw new ND4JIllegalStateException("Can't find mapped node [" + input + "]");
                    }


                    tNode.addInput(node, idx);
                } else
                    throw new RuntimeException("Unknown input passed in: [" + input + "]");

            } else {
                val id = reverseVertexMap.get(input);

                if (id == null)
                    throw new ND4JIllegalStateException("Unknown input: [" + input + "]");

                tNode.addInput(id);
            }
        }

        OpState opState = getOpStateFromNodeDef(tfNode, tfNode.getInputCount(), tNode, intermediateGraph.getVariableSpace());
        tNode.setOpState(opState);

        for (val index: tNode.getInputs()) {
            if (index.getNode() < 0)
                continue;

            val node = intermediateGraph.getNode(index.getNode());

            if (node != null)
                node.getOutputs().add(tNode.getId());
        }

        return tNode;
    }

    protected static TVariable importVariable(@NonNull NodeDef tfNode, @NonNull Map<String, TIndex> reverseVertexMap, int varId) {
        val variable = new TVariable();
        val attributes = tfNode.getAttrMap();
        List<Integer> dimensions = new ArrayList<>();

        reverseVertexMap.put(tfNode.getName(), TIndex.makeOf(varId, 0));

        boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
        boolean isVar = tfNode.getOp().startsWith("VariableV");
        boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

        variable.setName(tfNode.getName());
        variable.setId(varId);
        variable.setPlaceholder(isPlaceholder);

        int[] arrayShape = null;

        if (tfNode.getName().equalsIgnoreCase("mixed4b/concat_dim")) {
            log.debug("concat found!");
        }

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
            log.debug("Dtype: {}", tensor.getDtype());
            if (tensor.getDtype() == DataType.DT_FLOAT || tensor.getDtype() == DataType.DT_DOUBLE) {

                INDArray array = getNDArrayFromTensor(tensor);
                variable.setShape(array.shape());
                variable.setArray(array);
            } else {
                val shape = getShapeFromTensor(tensor);

                assert shape != null;
                assert shape.length > 0;

                variable.setShape(shape);
            }
        }

        if (!variable.isPlaceholder())
            log.debug("Variable: id: {}; name: {}; shape: {}", variable.getId(), variable.getName(), Arrays.toString(variable.getShape()));
        else
            log.debug("Placeholder shape: {}", Arrays.toString(variable.getShape()));


        return variable;
    }

    protected static void traverseList(@NonNull TGraph intermediateGraph, @NonNull List<NodeDef> tfNodesList, @NonNull Map<String, TIndex> reverseVertexMap,  Set<String> skipList, AtomicInteger varsCnt, AtomicInteger nodesCnt, int offset) {
        for (int e = offset; e < tfNodesList.size(); e++) {
            val tfNode = tfNodesList.get(e);

            if (skipList.contains(tfNode.getName().toLowerCase()))
                continue;

            skipList.add(tfNode.getName().toLowerCase());

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

            if (isConst || isVar || isPlaceholder) {
                val variable = importVariable(tfNode, reverseVertexMap, varsCnt.decrementAndGet());
                log.info("Adding var [{}:{}]", variable.getName(), variable.getId());

                intermediateGraph.getVariableSpace().addVariable(variable.getId(), variable);
            } else {
                int cCnt = nodesCnt.incrementAndGet();

                // operation node

                if (tfNode.getOp().equalsIgnoreCase("merge"))
                    continue;

                if (tfNode.getOp().equalsIgnoreCase("enter")) {
                    /*
                        on while/enter we'll open 2 scopes: 1st scope for condition, 2nd scope for loop body
                    */
                    val tNode = importWhileLoop(intermediateGraph, reverseVertexMap, e, skipList, tfNodesList, varsCnt, nodesCnt);
                    intermediateGraph.addNode(tNode);

                    continue;
                }

                log.info("Adding op [{}]", tfNode.getOp());

                val tNode = importNode(intermediateGraph, tfNode, reverseVertexMap, cCnt);

                log.debug("Node: {}", tNode);
                intermediateGraph.addNode(tNode);
            }
        }
    }

    /**
     * This method returns intermediate representation from TF GraphDef instance
     *
     * @return
     */
    public static TGraph importIntermediate(GraphDef tfGraph) {
        TGraph intermediateGraph = new TGraph();

        Map<String, TIndex> reverseVertexMap = new HashMap<>();

        val varsCnt = new AtomicInteger(0);
        val nodesCnt = new AtomicInteger(0);

        val skipCounter = new AtomicInteger(0);
        Set<String> skipList = new HashSet<>();
        val tfNodesList = tfGraph.getNodeList();

        // we're just starting our recursive fn here
        traverseList(intermediateGraph, tfNodesList, reverseVertexMap, skipList, varsCnt, nodesCnt, 0);

        return intermediateGraph;
    }

    protected static int[]  getShapeFromTensor(TensorProto tfTensor) {
        int[] arrayShape = null;

        if (tfTensor.getIntValCount() == 1) {
            int val = tfTensor.getIntVal(0);

            arrayShape = new int[]{val};

        } else if (tfTensor.getInt64ValCount() > 0) {
            arrayShape = new int[tfTensor.getInt64ValCount()];
            for (int e = 0; e < tfTensor.getInt64ValCount(); e++)
                arrayShape[e] = (int) tfTensor.getInt64Val(e);

        } else {
            // FIXME: INT bytebuffers should be converted to floating point
            if (tfTensor.getDtype() == DataType.DT_INT32) {
                val buffer = tfTensor.getTensorContent().asReadOnlyByteBuffer().order(ByteOrder.nativeOrder()).asIntBuffer();

                arrayShape = new int[buffer.capacity()];
                for (int e = 0; e < buffer.capacity(); e++)
                    arrayShape[e] = (int) buffer.get(e);
            } else if (tfTensor.getDtype() ==DataType.DT_INT64) {
                val buffer = tfTensor.getTensorContent().asReadOnlyByteBuffer().order(ByteOrder.nativeOrder()).asLongBuffer();

                arrayShape = new int[buffer.capacity()];
                for (int e = 0; e < buffer.capacity(); e++)
                    arrayShape[e] = (int) buffer.get(e);
            }

            log.debug("Array shape: {}", Arrays.toString(arrayShape));
        }

        return arrayShape;
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
                int dim = (int) tfTensor.getTensorShape().getDim(e).getSize();

                dimensions.add(dim);
            }
        }
        arrayShape = Ints.toArray(dimensions);

        if (tfTensor.getDtype() == DataType.DT_INT32 || tfTensor.getDtype() == DataType.DT_INT16 || tfTensor.getDtype() == DataType.DT_INT8) {
            // valueOf
            if (tfTensor.getIntValCount() == 1) {
                int val = tfTensor.getIntVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, (double) val);
                return array;
            } else if (tfTensor.getInt64ValCount() > 0) {
                double[] jArray = new double[tfTensor.getIntValCount()];
                for (int e = 0; e < tfTensor.getIntValCount(); e++) {
                    jArray[e] = (double) tfTensor.getIntVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            } else {
                // FIXME: INT bytebuffers should be converted to floating point
                throw new UnsupportedOperationException("To be implemented yet");
            }
        } else if (tfTensor.getDtype() == DataType.DT_FLOAT) {
            if (tfTensor.getFloatValCount() == 1) {
                float val = tfTensor.getFloatVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, (double) val);
                return array;
            } else if (tfTensor.getFloatValCount() > 0) {
                float[] jArray = new float[tfTensor.getFloatValCount()];
                for (int e = 0; e < tfTensor.getFloatValCount(); e++) {
                    jArray[e] = tfTensor.getFloatVal(e);
                }

                // FIXME: we're missing float[] signature
                INDArray array = Nd4j.create(ArrayUtil.toDoubles(jArray), arrayShape, 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0){

                long length = ArrayUtil.prodLong(arrayShape);
                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
                val fa = new float[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    fa[e] = fb.get(e);

                val array = Nd4j.create(fa, arrayShape, 'c', 0);
                //log.debug("SUM1: {}", array.sumNumber());
                //log.debug("Data: {}", Arrays.toString(array.data().asFloat()));
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_DOUBLE) {
            if (tfTensor.getDoubleValCount() == 1) {
                double val = tfTensor.getDoubleVal(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, val);
                return array;
            } else if (tfTensor.getDoubleValCount() > 0) {
                double[] jArray = new double[tfTensor.getDoubleValCount()];
                for (int e = 0; e < tfTensor.getDoubleValCount(); e++) {
                    jArray[e] =  tfTensor.getDoubleVal(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0) {
                long length = ArrayUtil.prodLong(arrayShape);
                // binary representation
                //DataBuffer buffer = Nd4j.createBuffer(tfTensor.getTensorContent().asReadOnlyByteBuffer(), DataBuffer.Type.FLOAT, (int) length);
                //INDArray array = Nd4j.createArrayFromShapeBuffer(buffer, Nd4j.getShapeInfoProvider().createShapeInformation(arrayShape, 'c'));

                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
                val da = new double[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    da[e] = fb.get(e);

                val array = Nd4j.create(da, arrayShape, 0, 'c');
                //log.debug("SUM1: {}", array.sumNumber());
                //log.debug("Data: {}", Arrays.toString(array.data().asFloat()));

                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_INT64) {
            if (tfTensor.getInt64ValCount() == 1) {
                double val = (double) tfTensor.getInt64Val(0);

                if (arrayShape == null || arrayShape.length == 0)
                    arrayShape = new int[]{1, 1};

                INDArray array = Nd4j.valueArrayOf(arrayShape, val);
                return array;
            } else if (tfTensor.getInt64ValCount() > 0)  {
                double[] jArray = new double[tfTensor.getInt64ValCount()];
                for (int e = 0; e < tfTensor.getInt64ValCount(); e++) {
                    jArray[e] =  (double) tfTensor.getInt64Val(e);
                }

                // TF arrays are always C
                INDArray array = Nd4j.create(jArray, arrayShape, 0, 'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0){
                // FIXME: INT bytebuffers should be converted to floating point
                throw new UnsupportedOperationException("To be implemented yet");
            }
        }  else {
            throw new UnsupportedOperationException("Unknown dataType found: [" + tfTensor.getDtype() + "]");
        }

        throw new RuntimeException("Wtf?");
    }

    protected static OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs) {
        return getOpStateFromNodeDef(tfNode, numInputs, null, null);
    }

    protected static OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs, TNode tNode, TVariableSpace variableSpace) {
        String lc = tfNode.getOp().toLowerCase();
        if (lc.equalsIgnoreCase("while"))
            log.info("While found");

        log.debug("Looking for [{}] op...", lc);
        if (numInputs > 0 && numInputs <= 2) {
            int opNum = Nd4j.getOpFactory().getOpNumIfExists(lc);

            if (opNum >= 0) {
                /*
                OpState opState = OpState.builder()
                        .opType(BaseOp.getOpType(Nd4j.getOpFactory().getOpByName(lc)))
                        .opNum(opNum)
                        .opName(lc)
                        .build();
                        */
                val type = BaseOp.getOpType(Nd4j.getOpFactory().getOpByName(lc));

                if (type != Op.Type.SHAPE && type != Op.Type.CUSTOM) {
                    val op = Nd4j.getOpFactory().getOpByName(lc);
                    OpState opState = OpState.builder()
                            .opType(type)
                            .extraArgs(op.extraArgs())
                            .opNum(opNum)
                            .opName(lc)
                            .build();

                    return opState;
                }
            }
        }

        OpState opState = OpState.builder()
                .opType(Op.Type.CUSTOM)
                .opNum(-1)
                .opName(tfNode.getOp())
                .build();

         if (lc.equalsIgnoreCase("conv2d")) {


             val aStrides = tfNode.getAttrOrThrow("strides");
             val tfStrides = aStrides.getList().getIList();
             val sY = tfStrides.get(1);
             val sX = tfStrides.get(2);

             val aPadding = tfNode.getAttrOrDefault("padding", null);

             val paddingMode = aPadding.getS().toStringUtf8();

             // we know that second input to conv2d is weights array
             val weightsIndex = tNode.getInputs().get(1);
             val variable = variableSpace.getVariable(weightsIndex);

             val kY = variable.getArray().size(0);
             val kX = variable.getArray().size(1);

             variable.setArray(variable.getArray().permute(3, 2, 0, 1).dup('c'));

             boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

             if (!isSameMode)
                 log.debug("Mode: {}", paddingMode);

            log.debug("Conv2D: k: [{}, {}]; s: [{}, {}]; padding: {}", kY, kX, sY, sX,  paddingMode);

             opState.setExtraBits(new int[] {kY, kX, sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0});
         } else if (lc.equalsIgnoreCase("avgpool") || lc.equalsIgnoreCase("maxpool")) {
             val aStrides = tfNode.getAttrOrThrow("strides");
             val tfStrides = aStrides.getList().getIList();
             val sY = tfStrides.get(1);
             val sX = tfStrides.get(2);

             val aKernels = tfNode.getAttrOrThrow("ksize");
             val tfKernels = aKernels.getList().getIList();

             val kY = tfKernels.get(1);
             val kX = tfKernels.get(2);

             val aPadding = tfNode.getAttrOrThrow("padding");

             val paddingMode = aPadding.getS().toStringUtf8().replaceAll("\"","");

             boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");

             if (!isSameMode)
                 log.debug("Mode: {}", paddingMode);

             log.debug("Pooling: k: [{},{}]; s: [{}, {}], padding: {}", kY, kX, sY, sX, aPadding);

             opState.setExtraBits(new int[] {kY.intValue(), kX.intValue(), sY.intValue(), sX.intValue(), 0, 0, 1, 1, isSameMode ? 1 : 0 });

         } else if (lc.equalsIgnoreCase("lrn")) {
             val aAlpha = tfNode.getAttrOrThrow("alpha");
             val aBeta = tfNode.getAttrOrThrow("beta");
             val aBias = tfNode.getAttrOrThrow("bias");
             val aDepth = tfNode.getAttrOrThrow("depth_radius");

             val alpha = aAlpha.getF();
             val beta = aBeta.getF();
             val bias = aBias.getF();
             val depth = aDepth.getF();


             opState.setExtraArgs(new Object[]{alpha, beta, bias, depth});
             log.debug("LRN: alpha: {}; beta: {}; bias: {}; depth: {};", alpha, beta, bias, depth);
         } else if (lc.equalsIgnoreCase("reshape")) {
             // in reshape operation we replace second input, and replace it with extra args
             log.debug("TNode inputs: {}", tNode.getInputs());
             val shapeIndex = tNode.getInputs().remove(1);
             val variable = variableSpace.getVariable(shapeIndex);

             assert variable != null;
             assert variable.getShape() != null;

             // we know that TF is always C order
             int[] args = ArrayUtils.add(variable.getShape(),  0, (int)'c');

             log.debug("Reshape node_{}, new shape: {}", tNode.getId(), Arrays.toString(args));

             // new shape goes here
             opState.setExtraBits(args);
         } else if (lc.equalsIgnoreCase("concat")) {
             log.debug("TNode inputs: {}", tNode.getInputs());
             TIndex dimIndex;
             int idx = -1;
             int cnt = 0;
             int concatDimension = 0;
             for (val index:tNode.getInputs()) {
                 log.debug("Trying to find node: [{}]", index);
                 val variable = variableSpace.getVariable(index);

                 // concat dimension is only possible
                 if (variable != null && variable.getId() < 0 && variable.getArray() == null) {
                     idx = cnt;
                     concatDimension = variable.getShape()[0];
                 }
                 cnt++;
             }

             if (idx < 0)
                 throw new ND4JIllegalStateException("Can't find dimension for concatenatiion");

             // deleting index of concat dimension
             tNode.getInputs().remove(idx);

             // if that's convolution graph, we should swap dimensions
             if (concatDimension == 3)
                 concatDimension = 1;

             opState.setExtraBits(new int[]{concatDimension});
             log.debug("Concat dimension: {}", concatDimension);
         }

         if (!Nd4j.getExecutioner().getCustomOperations().containsKey(lc))
             log.warn("Unknown op: [{}]", lc);
             //throw new ND4JIllegalStateException("Unknown operation requested: ["+ tfNode.getOp() +"]");

        return opState;
    }
}
