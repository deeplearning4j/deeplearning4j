package org.nd4j.imports.graphmapper.tf;

import com.google.common.primitives.Ints;
import com.google.protobuf.Message;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.converters.DifferentialFunctionClassHolder;
import org.nd4j.imports.graphmapper.BaseGraphMapper;
import org.nd4j.imports.graphmapper.ImportState;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.util.ArrayUtil;
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
    private Set<String> seenNodes = new HashSet<>();
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
        boolean isVar = nodeDef.getOp().startsWith("VariableV");
        return isVar;
    }

    @Override
    public boolean shouldSkip(NodeDef opType) {
        boolean isConst = opType.getOp().equalsIgnoreCase("const");
        boolean isVar = opType.getOp().startsWith("VariableV");
        boolean isPlaceholder = opType.getOp().startsWith("Placeholder");
        boolean endsWithRead = opType.getName().endsWith("/read");
        boolean isNoOp = opType.getOp().equalsIgnoreCase("NoOp");
        boolean isReductionIndices = opType.getOp().endsWith("/reduction_indices");

        return isConst || isVar || isPlaceholder || endsWithRead || isNoOp || isReductionIndices;
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


    @Override
    public Map<String, Integer> verticesForGraph(GraphDef graph, SameDiff sameDiff) {
        //map the names of the ndoes while accumulating the vertex ids
        //for each variable
        val variablesForGraph = variablesForGraph(graph);
        val indexMap = new HashMap<String,Integer>();
        for(Map.Entry<String,NodeDef> entry : variablesForGraph.entrySet()) {
            val var = sameDiff.var(entry.getKey(),getNDArrayFromTensor(entry.getKey(), entry.getValue(), graph));
            indexMap.put(entry.getKey(),var.getVertexId()[0]);
        }

        return indexMap;
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


    private String getNodeName(String name) {
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
    public Map<String, Pair<int[], int[]>> inputsAndOutputsForGraph(GraphDef graph, Map<String, Integer> nodeNameToVertexId) {
        val ret = new HashMap<String, Pair<int[], int[]>>(graph.getNodeCount());
        Map<String,List<Integer>> outputs = new HashMap<>();
        //map each node's outputs and inputs
        for(val node : graph.getNodeList()) {
            val nodeName = getNodeName(node.getName());
            //simultaneously collect the ids for inputs and outputs
            //incrementally building the list
            for(int i = 0; i < node.getInputCount(); i++) {
                val nodeInput = getNodeName(node.getInput(i));
                if(!outputs.containsKey(nodeName)) {
                    List<Integer> newInputs = new ArrayList<>();
                    val get = nodeNameToVertexId.get(nodeInput);
                    if(get != null)
                        newInputs.add(get);
                    else {
                        throw new ND4JIllegalStateException("Unable to map node " + nodeName + " no vertex id found!");
                    }
                    outputs.put(nodeName,newInputs);
                }
                else {
                    List<Integer> outputIds = outputs.get(nodeName);
                    val output = nodeNameToVertexId.get(nodeName);
                    if(output != null)
                        outputIds.add(nodeNameToVertexId.get(nodeInput));
                    else {
                        throw new ND4JIllegalStateException("Unable to map node " + nodeName + " no vertex id found!");
                    }

                }

            }
        }


        //collect the final result
        for(val entry : outputs.entrySet())  {
            val name = getNodeName(entry.getKey());
            val output = new int[] {nodeNameToVertexId.get(name)};
            int[] inputIds = Ints.toArray(entry.getValue());
            int[] outputIds = output;
            ret.put(name,Pair.of(inputIds,outputIds));



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
        if (shouldSkip(tfNode) || alreadySeen(tfNode)) {
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
            val vertexId = diff.getVariable(getName(tfNode)).getVertexId();
            diff.addAsPlaceHolder(vertexId);
        }
        else {
            val opName = tfNode.getOp();
            val differentialFunction = DifferentialFunctionClassHolder.getInstance().getOpWithTensorflowName(opName);
            try {
                val newInstance = differentialFunction.getClass().newInstance();
                val args = new DifferentialFunction[tfNode.getInputCount()];
                val indices = importState.getVertexIdMap().get(getNodeName(tfNode.getName()));

                for(int i = 0; i < tfNode.getInputCount(); i++) {
                    val name = getNodeName(tfNode.getInput(i));
                    val  initialVertexId = importState.getVertexIdMap().get(name);
                    int[] vertexIdKey = initialVertexId == null ? diff.getVariable(name).getVertexId() : initialVertexId.getRight();
                    if(vertexIdKey == null) {
                        throw new ND4JIllegalStateException("Unable set to set arg for op " + tfNode.getName());
                    }

                    DifferentialFunction func = diff.getFunctionForVertexId(vertexIdKey);
                    if(func == null) {
                        func =  diff.getVariable(name);
                    }

                    args[i] = func;

                    /**
                     * Note here that we are associating
                     * the output/result variable
                     * with its inputs and notifying
                     * the variable that it has a place holder argument
                     * it should resoolve before trying to execute
                     * anything.
                     */
                    if(diff.isPlaceHolder(func.getVertexId())) {
                        diff.putPlaceHolderForVertex(indices.getRight(),func.getVertexId());

                    }
                }




                val opStateEdge = getOpStateEdge(indices.getFirst(),indices.getSecond(),tfNode);
                diff.graph().addEdge(opStateEdge);
                diff.putFunction(indices.getRight(),newInstance);
                newInstance.setVertexId(indices.getRight());
                diff.associateFunctionsAsArgs(args,newInstance);
                newInstance.setSameDiff(importState.getSameDiff());

                newInstance.initFromTensorFlow(tfNode,diff,getAttrMap(tfNode),importState.getGraph());
                diff.putShapeForVertexId(indices.getRight(),newInstance.getResultShape());


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
