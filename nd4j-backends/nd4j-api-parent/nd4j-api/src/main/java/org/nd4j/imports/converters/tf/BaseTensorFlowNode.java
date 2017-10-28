package org.nd4j.imports.converters.tf;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.opstate.OpState;
import org.nd4j.graph.intermediate.TGraph;
import org.nd4j.graph.intermediate.TIndex;
import org.nd4j.graph.intermediate.TNode;
import org.nd4j.graph.intermediate.TVariableSpace;
import org.nd4j.imports.converters.ExternalNode;
import org.nd4j.linalg.api.ops.BaseOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.tensorflow.framework.NodeDef;

@Slf4j
public abstract class BaseTensorFlowNode implements ExternalNode<NodeDef> {

    protected BaseTensorFlowNode() {
        //
    }

    protected BaseTensorFlowNode(@NonNull Object nodeDef, @NonNull TGraph tGraph) {

    }

    /**
     * This method returns given TF node as TNode
     *
     * @return
     */
    @Override
    public abstract TNode asIntermediateRepresentation(@NonNull NodeDef node, @NonNull TGraph graph);

    /**
     * This method returns given TF node as ND4j Op
     *
     * @return
     */
    @Override
    public Op asExecutableOperation(@NonNull NodeDef node, @NonNull TGraph graph) {
        return null;
    }

    protected TNode buildBasicNode(@NonNull NodeDef tfNode, @NonNull TGraph intermediateGraph) {
        val nodeId = intermediateGraph.getCurrentNodeId();
        log.info("Adding reverse point [{}] -> [{}:0]", tfNode.getName(), nodeId);
        intermediateGraph.getReverseMap().put(tfNode.getName(), TIndex.makeOf(nodeId, 0));
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
                    Integer id = intermediateGraph.getReverseMap().get(split[0]).getNode();

                    tNode.addInput(id);
                } else if (split.length == 2) {
                    Integer lnode = intermediateGraph.getReverseMap().get(split[0]).getNode();
                    Integer idx = Integer.valueOf(split[1]);

                    if (lnode == null) {
                        log.error("Can't find mapped node [{}]", input);
                        throw new ND4JIllegalStateException("Can't find mapped node [" + input + "]");
                    }


                    tNode.addInput(lnode, idx);
                } else
                    throw new RuntimeException("Unknown input passed in: [" + input + "]");

            } else {
                val id = intermediateGraph.getReverseMap().get(input);

                if (id == null)
                    throw new ND4JIllegalStateException("TF Node [" + tfNode.getName() + "] refers to unknown input: [" + input + "]");

                tNode.addInput(id);
            }
        }

        tNode.setOpState(getOpStateFromNodeDef(tfNode, tfNode.getInputCount(), tNode, intermediateGraph.getVariableSpace()));

        for (val index: tNode.getInputs()) {
            if (index.getNode() < 0)
                continue;

            val lnode = intermediateGraph.getNode(index.getNode());

            if (lnode != null)
                lnode.getOutputs().add(tNode.getId());
        }

        return tNode;
    }


    protected OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs) {
        return getOpStateFromNodeDef(tfNode, numInputs, null, null);
    }

    protected OpState getOpStateFromNodeDef(NodeDef tfNode, int numInputs, TNode tNode, TVariableSpace variableSpace) {
        String lc = tfNode.getOp().toLowerCase();
        if (lc.equalsIgnoreCase("while"))
            log.info("While found");

        log.debug("Looking for [{}] op...", lc);
        if (numInputs > 0 && numInputs <= 2) {
            int opNum = Nd4j.getOpFactory().getOpNumIfExists(lc);

            if (opNum >= 0) {

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

        if (!Nd4j.getExecutioner().getCustomOperations().containsKey(lc))
            log.warn("Unknown op: [{}]", lc);
        //throw new ND4JIllegalStateException("Unknown operation requested: ["+ tfNode.getOp() +"]");

        return opState;
    }
}
