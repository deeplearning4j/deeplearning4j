package org.nd4j.imports.graphmapper.tf;

import com.google.common.primitives.Ints;
import com.google.protobuf.Message;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.weightinit.impl.ZeroInitScheme;
import org.tensorflow.framework.*;

import java.io.*;
import java.nio.ByteOrder;
import java.util.*;

/**
 * Map tensorflow graph protos
 * to the intermediate representation
 * for samediff.
 *
 * @author Adam Gibson
 */
@Slf4j
public class TFGraphMapper extends BaseGraphMapper<GraphDef,NodeDef,AttrValue,NodeDef> {
    private Set<String> seenNodes = new LinkedHashSet<>();
    public final static String VALUE_ATTR_KEY = "value";
    public final static String SHAPE_KEY = "shape";
    private static TFGraphMapper MAPPER_INSTANCE = new TFGraphMapper();

    //singleton
    private TFGraphMapper() {}

    /**
     * Singleton. Get the needed instance.
     * @return
     */
    public static TFGraphMapper getInstance() {
        return MAPPER_INSTANCE;
    }

    @Override
    public void dumpBinaryProtoAsText(InputStream inputFile, File outputFile) {
        try {
            GraphDef graphDef = GraphDef.parseFrom(inputFile);
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile,true));
            for(NodeDef node : graphDef.getNodeList()) {
                bufferedWriter.write(node.toString());
            }

            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public NodeDef getNodeWithNameFromGraph(GraphDef graph, String name) {
        for(int i = 0; i < graph.getNodeCount(); i++) {
            val node = graph.getNode(i);
            if(node.getName().equals(name))
                return node;
        }

        return null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isPlaceHolderNode(NodeDef node) {
        return node.getOp().startsWith("Placeholder");
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void dumpBinaryProtoAsText(File inputFile, File outputFile) {
        try {
            GraphDef graphDef = GraphDef.parseFrom(new BufferedInputStream(new FileInputStream(inputFile)));
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(outputFile,true));
            for(NodeDef node : graphDef.getNodeList()) {
                bufferedWriter.write(node.toString());
            }

            bufferedWriter.flush();
            bufferedWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public int[] getShapeFromAttr(AttrValue attr) {
        return shapeFromShapeProto(attr.getShape());
    }

    @Override
    public Map<String, AttrValue> getAttrMap(NodeDef nodeDef) {
        return nodeDef.getAttrMap();
    }

    @Override
    public String getName(NodeDef nodeDef) {
        return nodeDef.getName();
    }

    @Override
    public boolean alreadySeen(NodeDef nodeDef) {
        return seenNodes.contains(nodeDef.getName());
    }

    @Override
    public boolean isVariableNode(NodeDef nodeDef) {
        boolean isVar = nodeDef.getOp().startsWith("VariableV") || nodeDef.getOp().equalsIgnoreCase("const");
        return isVar;
    }

    @Override
    public boolean shouldSkip(NodeDef opType) {
        if(opType == null)
            return true;

        boolean endsWithRead = opType.getName().endsWith("/read");
        boolean isReductionIndices = opType.getOp().endsWith("/reduction_indices");
        return  endsWithRead  || isReductionIndices;
    }

    @Override
    public boolean hasShape(NodeDef nodeDef) {
        return nodeDef.containsAttr(SHAPE_KEY);
    }

    @Override
    public int[] getShape(NodeDef nodeDef) {
        return getShapeFromAttr(nodeDef.getAttrOrThrow(SHAPE_KEY));
    }

    @Override
    public INDArray getArrayFrom(NodeDef nodeDef, GraphDef graph) {
        return getNDArrayFromTensor(nodeDef.getName(),nodeDef, graph);
    }

    @Override
    public String getOpType(NodeDef nodeDef) {
        return nodeDef.getOp();
    }

    /**
     *
     * @param graphDef
     * @return
     */
    @Override
    public List<NodeDef> getNodeList(GraphDef graphDef) {
        return graphDef.getNodeList();
    }

    /**
     *
     * @param name the tensorflow or onnx name
     * @return
     */
    @Override
    public DifferentialFunction getMappedOp(String name) {
        return DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(name);
    }


    /**
     * Map a tensorflow node name
     * to the samediff equivalent
     * for import
     * @param name the name to change
     * @return the input tensorflow name
     */
    public String getNodeName(String name) {
        //tensorflow adds colons to the end of variables representing input index, this strips those off
        String ret = name;
        if(ret.startsWith("^"))
            ret = ret.substring(1);
        ret = ret.indexOf(':') >= 0 ? ret.substring(0,ret.indexOf(':')) : ret;
        if(ret.endsWith("/read")) {
            ret = ret.replace("/read","");
        }
        return ret;
    }

    @Override
    public Map<String, Pair<int[], int[]>> inputsAndOutputsForGraph(GraphDef graph) {
        val ret = new HashMap<String, Pair<int[], int[]>>(graph.getNodeCount());
        val nodeNameToVertexId = new HashMap<String,Integer>();
        int count = 1;
        for(val node : graph.getNodeList()) {
            if(shouldSkip(node))
                continue;
            val nodeName = getNodeName(node.getName());
            nodeNameToVertexId.put(nodeName,count++);
        }


        Map<String,List<Integer>> outputs = new HashMap<>();
        //map each node's outputs and inputs
        for(val node : graph.getNodeList()) {
            if(shouldSkip(node))
                continue;

            val nodeName = getNodeName(node.getName());
            //simultaneously collect the ids for inputs and outputs
            //incrementally building the list

            List<Integer> newInputs = new ArrayList<>(node.getInputCount());
            for(int i = 0; i < node.getInputCount(); i++) {
                val inputName = getNodeName(node.getInput(i));
                if(inputName.equals(nodeName))
                    continue;

                if(!outputs.containsKey(inputName)) {
                    if(nodeNameToVertexId.get(inputName) == null) {
                        throw new ND4JIllegalStateException("Node name " + inputName + " not found with vertex id!");
                    }
                    newInputs.add(nodeNameToVertexId.get(inputName));
                    outputs.put(inputName,newInputs);
                }
                else {

                    List<Integer> outputIds = outputs.get(inputName);
                    if(outputIds != null)
                        outputIds.add(nodeNameToVertexId.get(nodeName));
                    else {
                        throw new ND4JIllegalStateException("Unable to map node " + nodeName + " no vertex id found!");
                    }

                    newInputs.add(nodeNameToVertexId.get(inputName));


                }

            }


        }


        for(val node : graph.getNodeList()) {
            if (shouldSkip(node))
                continue;
            val nodeName = getNodeName(node.getName());
            int[] outputsForNode = !outputs.containsKey(nodeName) ? new int[0] : Ints.toArray(outputs.get(nodeName));
            if(!isVariableNode(node)) {
                val inputIndices = new int[node.getInputCount()];
                for(int i = 0; i < node.getInputCount(); i++) {
                    val inputName = getNodeName(node.getInput(i));
                    inputIndices[i] = nodeNameToVertexId.get(inputName);
                }

                if(outputsForNode.length < 1) {
                    outputsForNode = new int[]{nodeNameToVertexId.get(nodeName)};
                    ret.put(nodeName,Pair.of(inputIndices,outputsForNode));
                }

                ret.put(nodeName,Pair.of(inputIndices,outputsForNode));
            }

            else {
                if(node.getInputCount() < 1)
                    ret.put(nodeName,Pair.of(new int[]{nodeNameToVertexId.get(nodeName)},outputsForNode));
                if(outputsForNode.length < 1) {
                    outputsForNode = new int[]{nodeNameToVertexId.get(nodeName)};
                    ret.put(nodeName,Pair.of(new int[]{nodeNameToVertexId.get(nodeName)},outputsForNode));
                }

            }

        }

        return ret;
    }

    @Override
    public Map<String, NodeDef> variablesForGraph(GraphDef graphDef) {
        Map<String,NodeDef> ret = new LinkedHashMap<>();
        for(NodeDef nodeDef : graphDef.getNodeList()) {
            if(nodeDef.getName().endsWith("/read")) {
                continue;
            }


            val name = getNodeName(nodeDef.getName());
            ret.put(name,nodeDef);
        }

        return ret;
    }


    @Override
    public Message.Builder getNewGraphBuilder() {
        return GraphDef.newBuilder();
    }

    @Override
    public GraphDef parseGraphFrom(byte[] inputStream) throws IOException {
        return GraphDef.parseFrom(inputStream);
    }

    @Override
    public GraphDef parseGraphFrom(InputStream inputStream) throws IOException {
        return GraphDef.parseFrom(inputStream);
    }



    @Override
    public void mapNodeType(NodeDef tfNode, ImportState<GraphDef,NodeDef> importState) {
        if (shouldSkip(tfNode) || alreadySeen(tfNode) || isVariableNode(tfNode)) {
            return;
        }


        val diff = importState.getSameDiff();
        if (isVariableNode(tfNode)) {
            List<Integer> dimensions = new ArrayList<>();
            Map<String, AttrValue> attributes = getAttrMap(tfNode);
            if (attributes.containsKey(VALUE_ATTR_KEY)) {
                diff.var(getName(tfNode),getArrayFrom(tfNode,importState.getGraph()));
            }
            else if (attributes.containsKey(SHAPE_KEY)) {
                AttrValue shape = attributes.get(SHAPE_KEY);
                int[] shapeArr = getShapeFromAttr(shape);
                int dims = shapeArr.length;
                if (dims > 0) {
                    // even vector is 2d in nd4j
                    if (dims == 1)
                        dimensions.add(1);

                    for (int e = 0; e < dims; e++) {
                        // TODO: eventually we want long shapes :(
                        dimensions.add(getShapeFromAttr(shape)[e]);
                    }
                }
            }
        }

        else if(isPlaceHolder(tfNode)) {
            val vertexId = diff.getVariable(getName(tfNode));
            diff.addAsPlaceHolder(vertexId.getVarName());
        }
        else {
            val opName = tfNode.getOp();
            val differentialFunction = DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(opName);
            if(differentialFunction == null) {
                throw new ND4JIllegalStateException("No tensorflow op found for " + opName + " possibly missing operation class?");
            }
            try {
                val newInstance = differentialFunction.getClass().newInstance();
                val args = new SDVariable[tfNode.getInputCount()];
                val indices = importState.getVertexIdMap().get(getNodeName(tfNode.getName()));
                if(indices == null) {
                    throw new ND4JIllegalStateException("No indices found for " + tfNode.getName());
                }

                for(int i = 0; i < tfNode.getInputCount(); i++) {
                    val name = getNodeName(tfNode.getInput(i));
                    args[i] = diff.getVariable(name);

                    /**
                     * Note here that we are associating
                     * the output/result variable
                     * with its inputs and notifying
                     * the variable that it has a place holder argument
                     * it should resolve before trying to execute
                     * anything.
                     */
                    if(diff.isPlaceHolder( args[i].getVarName())) {
                        diff.putPlaceHolderForVariable(args[i].getVarName(), name);
                    }

                }

                //node
                /**
                 * Seems to not be linking last variable.
                 */
                if(indices.getFirst().length < 1 || indices.getSecond().length < 1)
                    return;


                val from = indices.getFirst();
                val to = indices.getSecond();

                diff.addArgsFor(args,newInstance);


                val varsList = new ArrayList<SDVariable>(diff.variables());
                //make sure variables are properly mapped
                if(diff.getVariable(TFGraphMapper.getInstance().getNodeName(tfNode.getName())) == null)
                    diff.var(TFGraphMapper.getInstance().getNodeName(tfNode.getName()),null,new ZeroInitScheme('f'));

                newInstance.setSameDiff(importState.getSameDiff());
                importState.getSameDiff().putFunctionForId(newInstance.getInstanceId(),newInstance);
                val outgoingArgs = new SDVariable[to.length];
                for(int i = 0; i < to.length; i++) {
                    outgoingArgs[i] = varsList.get(to[i] - 1);
                    if(outgoingArgs[i] == null) {
                        throw new ND4JIllegalStateException("Invalid variable for index: " + to[i]);
                    }
                }

                diff.addOutgoingFor(outgoingArgs,newInstance);
                newInstance.initFromTensorFlow(tfNode,diff,getAttrMap(tfNode),importState.getGraph());


            } catch (Exception e) {
                log.error("Failed with [{}]", opName);
                throw new RuntimeException(e);
            }

        }
    }

    @Override
    public DataBuffer.Type dataTypeForTensor(NodeDef tensorProto) {
        if(!tensorProto.containsAttr("dtype") && !tensorProto.containsAttr("Tidx") && !tensorProto.containsAttr("T"))
            return DataBuffer.Type.UNKNOWN;

        val type = tensorProto.containsAttr("dtype") ? tensorProto.getAttrOrThrow("dtype").getType()
                : tensorProto.containsAttr("T") ? tensorProto.getAttrOrThrow("T").getType() : tensorProto
                .getAttrOrThrow("Tidx").getType();
        switch(type) {
            case DT_DOUBLE: return DataBuffer.Type.DOUBLE;
            case DT_INT32:
            case DT_INT64: return DataBuffer.Type.INT;
            case DT_FLOAT: return DataBuffer.Type.FLOAT;
            case DT_BFLOAT16: return DataBuffer.Type.HALF;
            default: return DataBuffer.Type.UNKNOWN;
        }
    }



    @Override
    public String getAttrValueFromNode(NodeDef nodeDef, String key) {
        return nodeDef.getAttrOrThrow(key).getS().toStringUtf8();
    }

    @Override
    public int[] getShapeFromAttribute(AttrValue attrValue) {
        TensorShapeProto shape = attrValue.getShape();
        int[] ret = new int[shape.getDimCount()];
        for(int i = 0; i < ret.length; i++) {
            ret[i] = (int) shape.getDim(i).getSize();
        }
        return ret;
    }

    @Override
    public boolean isPlaceHolder(NodeDef nodeDef) {
        return nodeDef.getOp().startsWith("Placeholder");
    }

    @Override
    public  INDArray getNDArrayFromTensor(String tensorName, NodeDef node, GraphDef graph) {
        int[] arrayShape = null;
        List<Integer> dimensions = new ArrayList<>();
        //placeholder of some kind
        if(!node.getAttrMap().containsKey("value")) {
            return null;
        }
        val tfTensor = node.getAttrOrThrow("value").getTensor();
        // building shape first
        int dims = tfTensor.getTensorShape().getDimCount();
        if(dims == 1) {
            dimensions.add(1);
            dimensions.add( (int) Math.max(1,tfTensor.getTensorShape().getDim(0).getSize()));
        }
        else {
            for (int e = 0; e < dims; e++) {
                // TODO: eventually we want long shapes :(
                int dim = (int) tfTensor.getTensorShape().getDim(e).getSize();

                dimensions.add(dim);
            }
        }


        arrayShape = Ints.toArray(dimensions);

        if (tfTensor.getDtype() == DataType.DT_INT32 || tfTensor.getDtype() == DataType.DT_INT16 || tfTensor.getDtype() == DataType.DT_INT8) {
            // valueOf
            if (tfTensor.getIntValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getIntValCount() < 1)
                    return Nd4j.scalar(0.0);

                //should be scalar otherwise
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
            if (tfTensor.getFloatValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getFloatValCount() < 1)
                    return Nd4j.scalar(0.0);


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
                INDArray array = Nd4j.create(Nd4j.createBuffer(jArray), arrayShape,  'c');
                return array;
            } else if (tfTensor.getTensorContent().size() > 0){
                // binary representation
                val bb = tfTensor.getTensorContent().asReadOnlyByteBuffer();
                val fb = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
                val fa = new float[fb.capacity()];
                for (int e = 0; e < fb.capacity(); e++)
                    fa[e] = fb.get(e);

                val array = Nd4j.create(fa, arrayShape, 'c', 0);
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_DOUBLE) {
            if (tfTensor.getDoubleValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getDoubleValCount() < 1)
                    return Nd4j.scalar(0.0);

                double val = tfTensor.getDoubleVal(0);
                INDArray array = Nd4j.scalar(val);
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
                return array;
            }
        } else if (tfTensor.getDtype() == DataType.DT_INT64) {
            if (tfTensor.getInt64ValCount() == 1 || ArrayUtil.prod(arrayShape) == 1) {
                //straight zero case
                if(tfTensor.getDoubleValCount() < 1)
                    return Nd4j.scalar(0.0);

                double val = (double) tfTensor.getInt64Val(0);
                INDArray array = Nd4j.scalar(val);
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

        throw new ND4JIllegalStateException("Invalid method state");
    }

    @Override
    public int[] getShapeFromTensor(NodeDef tensorProto) {
        if(tensorProto.containsAttr("shape")) {
            return shapeFromShapeProto(tensorProto.getAttrOrThrow("shape").getShape());

        }
        //yet to be determined shape, or tied to an op where output shape is dynamic
        else if(!tensorProto.containsAttr("value")) {
            return null;

        }
        else
            return shapeFromShapeProto(tensorProto.getAttrOrThrow("value").getTensor().getTensorShape());
    }



    @Override
    public String getInputFromNode(NodeDef node, int index) {
        return node.getInput(index);
    }

    @Override
    public int numInputsFor(NodeDef nodeDef) {
        return nodeDef.getInputCount();
    }

    private int[] shapeFromShapeProto(TensorShapeProto tensorShapeProto) {
        int[] shape = new int[tensorShapeProto.getDimList().size()];
        for(int i = 0; i < shape.length; i++) {
            shape[i] = (int) tensorShapeProto.getDim(i).getSize();
        }

        return shape;
    }

}
