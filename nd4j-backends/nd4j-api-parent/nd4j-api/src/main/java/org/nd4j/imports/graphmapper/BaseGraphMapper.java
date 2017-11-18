package org.nd4j.imports.graphmapper;

import com.google.common.collect.Lists;
import com.google.common.primitives.Ints;
import com.google.protobuf.Message;
import com.google.protobuf.TextFormat;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.graph.intermediate.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.io.*;
import java.util.*;

/**
 * Base implementation for importing a graph
 * @param <GRAPH_TYPE> the type of graph
 * @param <NODE_TYPE> the type of node
 * @param <ATTR_TYPE> the attribute type
 */
@Slf4j
public abstract class BaseGraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE,TENSOR_TYPE> implements GraphMapper<GRAPH_TYPE,NODE_TYPE,ATTR_TYPE,TENSOR_TYPE> {
    /**
     *
     * @param graphFile
     * @return
     */
    @Override
    public  SameDiff importGraph(File graphFile) {
        GRAPH_TYPE def = null;
        try (FileInputStream fis = new FileInputStream(graphFile); BufferedInputStream bis = new BufferedInputStream(fis)) {
            def = parseGraphFrom(bis);
        } catch (Exception e) {
            try (FileInputStream fis2 = new FileInputStream(graphFile); BufferedInputStream bis2 = new BufferedInputStream(fis2); BufferedReader reader = new BufferedReader(new InputStreamReader(bis2))) {
                Message.Builder builder = getNewGraphBuilder();

                StringBuilder str = new StringBuilder();
                String line = null;
                while ((line = reader.readLine()) != null) {
                    str.append(line);//.append("\n");
                }

                TextFormat.getParser().merge(str.toString(), builder);
                def = (GRAPH_TYPE) builder.build();
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
    @Override
    public SameDiff importGraph(GRAPH_TYPE tfGraph) {
        SameDiff diff = SameDiff.create();
        ImportState<GRAPH_TYPE,TENSOR_TYPE> importState = new ImportState<>();
        importState.setNodeCount(0);
        importState.setSameDiff(diff);
        importState.setGraph(tfGraph);
        Map<String, TENSOR_TYPE> variablesForGraph = variablesForGraph(tfGraph);
        importState.setVariables(variablesForGraph);
        for(Map.Entry<String,TENSOR_TYPE> entry : variablesForGraph.entrySet()) {
            importState.getSameDiff().var(entry.getKey(),getNDArrayFromTensor(entry.getValue()));

        }

        val tfNodesList = getNodeList(tfGraph);
        for (NODE_TYPE tfNode : tfNodesList) {
            mapNodeType(tfNode,importState);
        }

        return diff;
    }


    /**
     * This method returns intermediate representation from TF GraphDef instance
     *
     * @return
     */
    public  TGraph importIntermediate(GRAPH_TYPE tfGraph) {
        TGraph intermediateGraph = new TGraph();
        val tfNodesList = getNodeList(tfGraph);

        // we're just starting our recursive fn here
        traverseList(intermediateGraph, tfNodesList, 0);

        return intermediateGraph;
    }

    protected  TIndex indexByName(@NonNull TGraph graph, @NonNull String value) {
        if (value.contains(":")) {
            val split = value.split(":");
            Integer lnode = graph.getReverseMap().get(split[0]).getNode();
            Integer idx = Integer.valueOf(split[1]);

            return TIndex.makeOf(lnode, idx);
        } else {
            Integer lnode = graph.getReverseMap().get(value).getNode();
            return TIndex.makeOf(lnode);
        }
    }

    protected TOp importWhileLoop(TGraph intermediateGraph, int startPosition, List<NODE_TYPE> nodes) {
        val uniqueId = java.util.UUID.randomUUID().toString();

        val scopeCondition = new TScope(intermediateGraph.getNewNodeId(), "scopeCondition_" + uniqueId);
        val scopeLoop = new TScope(intermediateGraph.getNewNodeId(), "scopeLoop_" + uniqueId);

        intermediateGraph.addScope(scopeCondition);
        intermediateGraph.addScope(scopeLoop);

        val whileNode = TOp.builder().id(intermediateGraph.getNewNodeId())
                .opName("while")
                .name("whileLoop_" + uniqueId)
                .opNum(0)
                .opState(OpState.builder().opName("while").opNum(0).opType(Op.Type.LOOP).build())
                .build();

        log.info("WHILE id: {}", uniqueId);
        log.info("Adding 2 new scopes for WHILE {}", whileNode.getId());


        /**
         * Plan is simple:
         * 1) we read all declarations of variables used within loop
         * 2) we set up conditional scope
         * 3) we set up body scope
         * 4) ???
         * 5) PROFIT!
         */



        // parsing declarations first. they all come as Enter ops
        val whileInputs = new ArrayList<TIndex>();
        int enterCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!getOpType(tfNode).equalsIgnoreCase("enter")) {
                //intermediateGraph.getSkipSet().add(getName(tfNode));
                break;
            }

            intermediateGraph.getSkipSet().add(getName(tfNode));

            // enter should have only 1 input, but let's keep loop here.
            for (int e = 0; e < numInputsFor(tfNode); e++) {
                val input = getInputFromNode(tfNode,e);
                //val idx = intermediateGraph.getReverseMap().get(input);
                val idx = indexByName(intermediateGraph, input);
                log.info("Enter mapping [{}] to [{}]", input, idx);

                // mapping this
                whileInputs.add(idx);
            }

            intermediateGraph.getReverseMap().put(getName(tfNode), TIndex.makeOf(whileNode.getId(), enterCnt++));
        }

        whileInputs.add(TIndex.makeOf(scopeCondition.getId()));
        whileInputs.add(TIndex.makeOf(scopeLoop.getId()));
        whileNode.setInputs(whileInputs);

        // now we're skipping Merge step, since we've already captured variables at Enter step
        int mergedCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (!getOpType(tfNode).equalsIgnoreCase("merge")) {
                break;
            }

            intermediateGraph.getSkipSet().add(getName(tfNode));

            //localReverseMap.put(getName(tfNode), TIndex.makeOf(tNode.getId(), mergedCnt));
            intermediateGraph.getReverseMap().put(getName(tfNode), TIndex.makeOf(whileNode.getId(), mergedCnt++));
        }


        // now, we're adding conditional scope
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            // we're parsing up to condition
            if (getOpType(tfNode).equalsIgnoreCase("LoopCond")) {
                intermediateGraph.getSkipSet().add(getName(tfNode));
                startPosition++;
                break;
            }

            boolean isConst = getOpType(tfNode).equalsIgnoreCase("const");
            boolean isVar = getOpType(tfNode).startsWith("VariableV");
            boolean isPlaceholder = getOpType(tfNode).startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = importVariable(tfNode, intermediateGraph.getReverseMap(), intermediateGraph.getNewVariableId());
                log.info("Adding condition var [{}:{}]", var.getName(), var.getId());

                intermediateGraph.getVariableSpace().addVariable(var.getId(), var);
            } else {
                log.info("starting on [{}]: {}", getName(tfNode), getOpType(tfNode));

                val scopedNode = importNode(intermediateGraph, tfNode, intermediateGraph.getNewNodeId());
                scopedNode.setScoped(true);
                scopedNode.setScopeId(scopeCondition.getId());
                scopedNode.setScopeName(scopeCondition.getName());

                scopeCondition.addNode(scopedNode);
            }

            intermediateGraph.getSkipSet().add(getName(tfNode));
        }



        // time to skip some Switch calls
        int switchCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            // we're parsing up to condition
            if (!getOpType(tfNode).equalsIgnoreCase("Switch"))
                break;

            intermediateGraph.getReverseMap().put(getName(tfNode), TIndex.makeOf(whileNode.getId(), switchCnt++));
            intermediateGraph.getSkipSet().add(getName(tfNode));
        }

        // now we're parsing Identity step
        int identityCnt = 0;
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);


            if (!getOpType(tfNode).equalsIgnoreCase("Identity")) {
                break;
            }

            val scopedNode = importNode(intermediateGraph, tfNode, intermediateGraph.getNewNodeId());
            scopedNode.setScopeId(scopeLoop.getId());
            scopedNode.setScopeName(scopeLoop.getName());

            // we overwrite inputs here, because that's always mapping to the While scope operands
            scopedNode.setInputs(Lists.newArrayList(TIndex.makeOf(whileNode.getId(), identityCnt++)));

            scopeLoop.addNode(scopedNode);

            intermediateGraph.getSkipSet().add(getName(tfNode));
        }


        // parsing body scope
        for (; startPosition < nodes.size(); startPosition++) {
            val tfNode = nodes.get(startPosition);

            if (intermediateGraph.getSkipSet().contains(getName(tfNode))) {
                log.info("Skipping: {}", getName(tfNode));
                continue;
            }

            if (getOpType(tfNode).equalsIgnoreCase("NextIteration")) {
//                intermediateGraph.getSkipSet().add(getName(tfNode));
                break;
            }

            if (intermediateGraph.getSkipSet().contains(getName(tfNode))) {
                log.info("Skipping: {}", getName(tfNode));
                continue;
            }



            boolean isConst = getOpType(tfNode).equalsIgnoreCase("const");
            boolean isVar = getOpType(tfNode).startsWith("VariableV");
            boolean isPlaceholder = getOpType(tfNode).startsWith("Placeholder");


            if (isConst || isVar || isPlaceholder) {
                val var = importVariable(tfNode, intermediateGraph.getReverseMap(), intermediateGraph.getNewVariableId());
                log.info("Adding body var [{}:{}]", var.getName(), var.getId());

                intermediateGraph.getVariableSpace().addVariable(var.getId(), var);
            } else {
                log.info("starting on [{}]: {}", getName(tfNode), getOpType(tfNode));

                boolean isNewLoop = false;
                if (getOpType(tfNode).equalsIgnoreCase("enter")) {
                    val frame_name = getAttrValueFromNode(tfNode,"frame_name");
                    if (!intermediateGraph.getKnownScopes().contains(frame_name)) {
                        intermediateGraph.getKnownScopes().add(frame_name);
                        isNewLoop = true;
                    }
                }

                if (isNewLoop) {
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

            intermediateGraph.getSkipSet().add(getName(tfNode));
        }

        val returnOp = TOp.builder()
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

            if (!getOpType(tfNode).equalsIgnoreCase("NextIteration"))
                break;

            intermediateGraph.getSkipSet().add(getName(tfNode));

            val inputName = getInputFromNode(tfNode,0);
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

            if (!getOpType(tfNode).equalsIgnoreCase("Exit")) {
                //intermediateGraph.getSkipSet().add(getName(tfNode));
                break;
            }

            intermediateGraph.getSkipSet().add(getName(tfNode));
            intermediateGraph.getReverseMap().put(getName(tfNode), TIndex.makeOf(whileNode.getId(), exitCnt++));
        }


        log.info("-------------------------------------------");

        return whileNode;
    }


    /**
     * This method converts given TF
     * @param tfGraph
     * @return
     */
    @Override
    public  SameDiff mapGraph(GRAPH_TYPE tfGraph) {
        SameDiff diff = SameDiff.create();

        Set<String> skipList = new HashSet<>();
        val tfNodesList = getNodeList(tfGraph);
        for (NODE_TYPE tfNode : tfNodesList) {
            log.debug("Node opName: {}; Op: {};", getName(tfNode), getOpType(tfNode));

            if (getOpType(tfNode).equalsIgnoreCase("enter") || (skipList.contains(getName(tfNode)))) {
                continue;
            }


            boolean isConst = getOpType(tfNode).equalsIgnoreCase("const");
            boolean isVar = getOpType(tfNode).startsWith("VariableV");
            boolean isPlaceholder = getOpType(tfNode).startsWith("Placeholder");

            Map<String, ATTR_TYPE> attributes = getAttrMap(tfNode);
            if (isConst || isVar || isPlaceholder) {
                if (attributes.containsKey(valueKey())) {
                    // value of?
                    ATTR_TYPE value = attributes.get(valueKey());

                    //DataType opType = value.
                    TENSOR_TYPE tensor = getTensorFrom(value);
                    log.debug("Dtype: {}", dataTypeForTensor(tensor));

                    INDArray array = getNDArrayFromTensor(tensor);
                    SDVariable var = diff.var(getName(tfNode), array);


                }

                else  if (attributes.containsKey(shapeKey())) {
                    ATTR_TYPE shape = attributes.get(shapeKey());
                    int[] shapeArr = getShapeFromAttr(shape);
                    SDVariable sdVariable = diff.var(getName(tfNode), shapeArr);

                }
            }
        }
        return diff;
    }



    protected  TOp importNode(@NonNull TGraph intermediateGraph, @NonNull NODE_TYPE tfNode, int nodeId) {
        val tNode = asIntermediate(tfNode, intermediateGraph, getAttrMap(tfNode));
        return tNode;
    }

    protected  TVariable importVariable(@NonNull NODE_TYPE tfNode, @NonNull Map<String, TIndex> reverseVertexMap, int varId) {
        val variable = new TVariable();
        val attributes = getAttrMap(tfNode);
        List<Integer> dimensions = new ArrayList<>();

        reverseVertexMap.put(getName(tfNode), TIndex.makeOf(varId, 0));

        variable.setName(getName(tfNode));
        variable.setId(varId);
        variable.setPlaceholder(isPlaceHolder(tfNode));

        int[] arrayShape = null;



        if (attributes.containsKey("shape")) {
            ATTR_TYPE shape = attributes.get("shape");
            int[] shapeArr = getShapeFromAttribute(shape);
            int dims = shapeArr.length;
            if (dims > 0) {

                // even vector is 2d in nd4j
                if (dims == 1)
                    dimensions.add(1);

                for (int e = 0; e < dims; e++) {
                    // TODO: eventually we want long shapes :(
                    dimensions.add(shapeArr[e]);
                }
            }

            arrayShape = Ints.toArray(dimensions);

            variable.setShape(arrayShape);
        }

        if (attributes.containsKey(valueKey())) {
            // value of?
            ATTR_TYPE value = attributes.get(valueKey());

            //DataType type = value.

            TENSOR_TYPE tensor = getTensorFrom(value);
            DataBuffer.Type  dType = dataTypeForTensor(tensor);
            log.debug("Dtype: {}", dType);
            if (validTensorDataType(tensor)) {
                INDArray array = getNDArrayFromTensor(tensor);
                variable.setShape(array.shape());
                variable.setArray(array);
            }

            else {
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
            log.debug("Variable: id: {}; opName: {}; shape: {}", variable.getId(), variable.getName(), Arrays.toString(variable.getShape()));
        else
            log.debug("Placeholder shape: {}", Arrays.toString(variable.getShape()));


        return variable;
    }

    @Override
    public boolean validTensorDataType(TENSOR_TYPE tensorType) {
        return dataTypeForTensor(tensorType) != DataBuffer.Type.UNKNOWN;
    }

    protected void traverseList(@NonNull TGraph intermediateGraph, @NonNull List<NODE_TYPE> tfNodesList, int offset) {
        for (int e = offset; e < tfNodesList.size(); e++) {
            val tfNode = tfNodesList.get(e);

            if (intermediateGraph.getSkipSet().contains(getName(tfNode)))
                continue;

            intermediateGraph.getSkipSet().add(getName(tfNode));

            boolean isConst = getOpType(tfNode).equalsIgnoreCase("const");
            boolean isVar = getOpType(tfNode).startsWith("VariableV");
            boolean isPlaceholder = getOpType(tfNode).startsWith("Placeholder");

            if (isConst || isVar || isPlaceholder) {
                val variable = importVariable(tfNode, intermediateGraph.getReverseMap(), intermediateGraph.getNewVariableId());
                log.info("Adding var [{}:{}]", variable.getName(), variable.getId());

                intermediateGraph.getVariableSpace().addVariable(variable.getId(), variable);
            } else {
                int cCnt = intermediateGraph.getNewNodeId();

                // operation node
                if (getOpType(tfNode).equalsIgnoreCase("NoOp"))
                    continue;

                if (getOpType(tfNode).equalsIgnoreCase("merge"))
                    continue;

                boolean isNewLoop = false;
                if (getOpType(tfNode).equalsIgnoreCase("enter")) {
                    val frame_name = getAttrValueFromNode(tfNode,"frame_name");
                    if (!intermediateGraph.getKnownScopes().contains(frame_name)) {
                        intermediateGraph.getKnownScopes().add(frame_name);
                        isNewLoop = true;
                    }
                }

                if (isNewLoop) {
                    log.info("NEW LOOP --------------------");
                    /*
                        on while/enter we'll open 2 scopes: 1st scope for condition, 2nd scope for loop body
                    */
                    val tNode = importWhileLoop(intermediateGraph, e, tfNodesList);
                    intermediateGraph.addNode(tNode);

                    continue;
                }

                log.info("Adding op [{}]", getOpType(tfNode));

                val tNode = importNode(intermediateGraph, tfNode, cCnt);

                log.info("Node: {}", tNode);
                intermediateGraph.addNode(tNode);
            }
        }
    }
}
