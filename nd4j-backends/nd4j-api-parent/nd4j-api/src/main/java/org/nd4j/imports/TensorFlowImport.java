package org.nd4j.imports;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.Ints;
import com.google.protobuf.TextFormat;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.opstate.NDArrayVertex;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDGraph;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.graph.intermediate.*;
import org.nd4j.imports.converters.TensorFlowMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.*;

import java.io.*;
import java.nio.ByteOrder;
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
                .variableMap(Maps.<String, SDVariable>newHashMap())
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

                SDVariable varInformation = SDVariable.builder()
                        .varName(tfNode.getName())
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
                    variable.getSameDiff().associateArrayWithVariable(array,variable);
                }

                diff.addVariable(variable);
                graph.addVertex(vertex);
            } else {
                // operation node

                /*

                   SDVariable varInformation = SDVariable.builder()
                        .varName(tfNode.getName())
                NDArrayInformation varInformation = NDArrayInformation.builder()
                        .id(tfNode.getName())
                        .build();

                NDArrayVertex vertex = new NDArrayVertex(diff,++nodesCnt, 0,varInformation);
                graph.addVertex(vertex);

                OpState opState = getOpStateFromNodeDef(tfNode, tfNode.getInputCount());
                opState.setResults(new SDVariable[]{varInformation});

                reverseVertexMap.put(tfNode.getName(), nodesCnt);


                for (int e = 0; e < tfNode.getInputCount(); e++) {
                    String input = tfNode.getInput(e);

                    Integer id = reverseVertexMap.get(input);

                    if (id == null)
                        throw new ND4JIllegalStateException("Unknown input: [" + input + "]");

                    graph.addEdge(new int[]{id}, new int[]{nodesCnt}, opState, true);
                }
                */
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

    protected static TNode importWhileLoop(TGraph intermediateGraph, int startPosition, List<NodeDef> nodes) {
        val uniqueId = java.util.UUID.randomUUID().toString();

        val scopeCondition = new TScope(intermediateGraph.getNewNodeId(), "scopeCondition_" + uniqueId);
        val scopeLoop = new TScope(intermediateGraph.getNewNodeId(), "scopeLoop_" + uniqueId);

        intermediateGraph.addScope(scopeCondition);
        intermediateGraph.addScope(scopeLoop);

        val whileNode = TNode.builder().id(intermediateGraph.getNewNodeId())
                .opName("while")
                .name("whileLoop_" + uniqueId)
                .opNum(0)
                .opState(OpState.builder().opName("while").opNum(0).opType(Op.Type.LOOP).build())
                .build();

        log.info("Adding 2 new scopes for WHILE {}", whileNode.getId());


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
        val whileInputs = new ArrayList<TIndex>();
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("enter")) {
                //intermediateGraph.getSkipSet().add(tfNode.getName());
                break;
            }

//            if (intermediateGraph.getSkipSet().contains(tfNode.getName()))
//                continue;

            intermediateGraph.getSkipSet().add(tfNode.getName());

            for (int e = 0; e < tfNode.getInputCount(); e++) {
                val input = tfNode.getInput(e);
                val idx = intermediateGraph.getReverseMap().get(input);
                log.info("Mapping [{}] to [{}]", input, idx);

                // mapping this
                whileInputs.add(idx);
            }
        }
        whileInputs.add(TIndex.makeOf(scopeCondition.getId()));
        whileInputs.add(TIndex.makeOf(scopeLoop.getId()));
        whileNode.setInputs(whileInputs);

        // now we're skipping Merge step, since we've already captured variables at Enter step
        int mergedCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("merge")) {
                break;
            }

            intermediateGraph.getSkipSet().add(tfNode.getName());

            //localReverseMap.put(tfNode.getName(), TIndex.makeOf(tNode.getId(), mergedCnt));
            intermediateGraph.getReverseMap().put(tfNode.getName(), TIndex.makeOf(whileNode.getId(), mergedCnt++));
        }


        // now, we're adding conditional scope
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            // we're parsing up to condition
            if (tfNode.getOp().equalsIgnoreCase("LoopCond")) {
                intermediateGraph.getSkipSet().add(tfNode.getName());
                startPosition++;
                break;
            }

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = importVariable(tfNode, intermediateGraph.getReverseMap(), intermediateGraph.getNewVariableId());
                log.info("Adding condition var [{}:{}]", var.getName(), var.getId());

                intermediateGraph.getVariableSpace().addVariable(var.getId(), var);
            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());

                val scopedNode = importNode(intermediateGraph, tfNode, intermediateGraph.getNewNodeId());
                scopedNode.setScoped(true);
                scopedNode.setScopeId(scopeCondition.getId());
                scopedNode.setScopeName(scopeCondition.getName());

                scopeCondition.addNode(scopedNode);
            }

            intermediateGraph.getSkipSet().add(tfNode.getName());
        }



        // time to skip some Switch calls
        int switchCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            // we're parsing up to condition
            if (!tfNode.getOp().equalsIgnoreCase("Switch"))
                break;

            intermediateGraph.getReverseMap().put(tfNode.getName(), TIndex.makeOf(whileNode.getId(), switchCnt++));
            intermediateGraph.getSkipSet().add(tfNode.getName());
        }

        // now we're parsing Identity step
        int identityCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);


            if (!tfNode.getOp().equalsIgnoreCase("Identity")) {
                break;
            }

            val scopedNode = importNode(intermediateGraph, tfNode, intermediateGraph.getNewNodeId());
            scopedNode.setScopeId(scopeLoop.getId());
            scopedNode.setScopeName(scopeLoop.getName());

            // we overwrite inputs here, because that's always mapping to the While scope operands
            scopedNode.setInputs(Lists.newArrayList(TIndex.makeOf(whileNode.getId(), identityCnt++)));

            scopeLoop.addNode(scopedNode);

            intermediateGraph.getSkipSet().add(tfNode.getName());
        }


        // parsing body scope
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (intermediateGraph.getSkipSet().contains(tfNode.getName())) {
                log.info("Skipping: {}", tfNode.getName());
                continue;
            }

            if (tfNode.getOp().equalsIgnoreCase("NextIteration")) {
//                intermediateGraph.getSkipSet().add(tfNode.getName());
                break;
            }

            if (intermediateGraph.getSkipSet().contains(tfNode.getName())) {
                log.info("Skipping: {}", tfNode.getName());
                continue;
            }



            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = importVariable(tfNode, intermediateGraph.getReverseMap(), intermediateGraph.getNewVariableId());
                log.info("Adding body var [{}:{}]", var.getName(), var.getId());

                intermediateGraph.getVariableSpace().addVariable(var.getId(), var);
            } else {
                log.info("starting on [{}]: {}", tfNode.getName(), tfNode.getOp());

                if (tfNode.getOp().equalsIgnoreCase("enter")) {
                    log.info("NEW LOOP ----------------------------------------");
                    val scopedWhile = importWhileLoop(intermediateGraph, startPosition, nodes);
                    scopedWhile.setScoped(true);
                    scopedWhile.setScopeId(scopeLoop.getId());
                    scopedWhile.setScopeName(scopeLoop.getName());

                    scopeLoop.addNode(scopedWhile);

                    log.info("END LOOP ----------------------------------------");
                } else {
                    val scopedNode = importNode(intermediateGraph, tfNode, intermediateGraph.getNewNodeId());
                    scopedNode.setScoped(true);
                    scopedNode.setScopeId(scopeLoop.getId());
                    scopedNode.setScopeName(scopeLoop.getName());

                    scopeLoop.addNode(scopedNode);
                }
            }

            intermediateGraph.getSkipSet().add(tfNode.getName());
        }

        val returnOp = TNode.builder()
                .opState(OpState.builder()
                        .opType(Op.Type.RETURN)
                        .opNum(40)
                        .opName("return")
                        .build())
                .name("whileReturn_" + uniqueId)
                .id(intermediateGraph.getNewNodeId())
                .opName("return")
                .opNum(40)
                .scoped(true)
                .scopeId(scopeLoop.getId())
                .scopeName(scopeLoop.getName())
                .build();

        val returnInputs = new ArrayList<TIndex>();
        val returnOutputs = new ArrayList<Integer>();
        // mapping NextIterations, to Return op
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("NextIteration"))
                break;

            intermediateGraph.getSkipSet().add(tfNode.getName());

            val inputName = tfNode.getInput(0);
            val input = intermediateGraph.getReverseMap().get(inputName);
            returnInputs.add(input);
            returnOutputs.add(whileNode.getId());
        }
        returnOp.setInputs(returnInputs);
        returnOp.setOutputs(returnOutputs);
        scopeLoop.addNode(returnOp);

        // we should also map While/Exit to libnd4j while
        int exitCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!tfNode.getOp().equalsIgnoreCase("Exit")) {
                //intermediateGraph.getSkipSet().add(tfNode.getName());
                break;
            }

            intermediateGraph.getSkipSet().add(tfNode.getName());

            intermediateGraph.getReverseMap().put(tfNode.getName(), TIndex.makeOf(whileNode.getId(), exitCnt++));
        }


        log.info("-------------------------------------------");

        return whileNode;
    }



    protected static TNode importNode(@NonNull TGraph intermediateGraph, @NonNull NodeDef tfNode, int nodeId) {

        val tNode = TensorFlowMapper.getInstance().asIntermediate(tfNode, intermediateGraph);

        return tNode;
    }

    protected static TVariable importVariable(@NonNull NodeDef tfNode, @NonNull Map<String, TIndex> reverseVertexMap, int varId) {
        if (tfNode.getName().equalsIgnoreCase("while/Less/y"))
        //if (tfNode.getName().equalsIgnoreCase("while/Sum"))
            log.debug("wow");

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

        if (tfNode.getName().equalsIgnoreCase("while/Const"))
            log.debug("");

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
            if (tensor.getDtype() == DataType.DT_FLOAT || tensor.getDtype() == DataType.DT_DOUBLE || tensor.getDtype() == DataType.DT_INT32 || tensor.getDtype() == DataType.DT_INT64) {

                INDArray array = getNDArrayFromTensor(tensor);
                variable.setShape(array.shape());
                variable.setArray(array);
            } else {
                val shape = getShapeFromTensor(tensor);

                assert shape != null;
                assert shape.length > 0;

                // in most of cases this loop will fix scalars. i.e shapes [0, 1] or [1, 0]
                for (int e = 0; e < shape.length; e++)
                    if (shape[e] == 0)
                        shape[e] = 1;

                variable.setShape(shape);
            }
        }

        if (!variable.isPlaceholder())
            log.debug("Variable: id: {}; name: {}; shape: {}", variable.getId(), variable.getName(), Arrays.toString(variable.getShape()));
        else
            log.debug("Placeholder shape: {}", Arrays.toString(variable.getShape()));


        return variable;
    }

    protected static void traverseList(@NonNull TGraph intermediateGraph, @NonNull List<NodeDef> tfNodesList, int offset) {
        for (int e = offset; e < tfNodesList.size(); e++) {
            val tfNode = tfNodesList.get(e);

            if (intermediateGraph.getSkipSet().contains(tfNode.getName()))
                continue;

            intermediateGraph.getSkipSet().add(tfNode.getName());

            boolean isConst = tfNode.getOp().equalsIgnoreCase("const");
            boolean isVar = tfNode.getOp().startsWith("VariableV");
            boolean isPlaceholder = tfNode.getOp().startsWith("Placeholder");

            if (isConst || isVar || isPlaceholder) {
                val variable = importVariable(tfNode, intermediateGraph.getReverseMap(), intermediateGraph.getNewVariableId());
                log.info("Adding var [{}:{}]", variable.getName(), variable.getId());

                intermediateGraph.getVariableSpace().addVariable(variable.getId(), variable);
            } else {
                int cCnt = intermediateGraph.getNewNodeId();

                // operation node
                if (tfNode.getOp().equalsIgnoreCase("NoOp"))
                    continue;

                if (tfNode.getOp().equalsIgnoreCase("merge"))
                    continue;

                if (tfNode.getOp().equalsIgnoreCase("enter")) {
                    /*
                        on while/enter we'll open 2 scopes: 1st scope for condition, 2nd scope for loop body
                    */
                    val tNode = importWhileLoop(intermediateGraph, e, tfNodesList);
                    intermediateGraph.addNode(tNode);

                    continue;
                }

                log.info("Adding op [{}]", tfNode.getOp());

                val tNode = importNode(intermediateGraph, tfNode, cCnt);

                log.info("Node: {}", tNode);
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
        traverseList(intermediateGraph, tfNodesList, 0);

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
                //throw new UnsupportedOperationException("To be implemented yet");
                long length = ArrayUtil.prodLong(arrayShape);
                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asIntBuffer();
                val fa = new float[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    fa[e] = (float) fb.get(e);

                val array = Nd4j.create(fa, arrayShape, 'c', 0);
                //log.debug("SUM1: {}", array.sumNumber());
                //log.debug("Data: {}", Arrays.toString(array.data().asFloat()));
                return array;
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
                val fb = bb.order(ByteOrder.nativeOrder()).asDoubleBuffer();
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

}
