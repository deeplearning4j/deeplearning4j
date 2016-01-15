package org.deeplearning4j.nn.graph;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.nodes.GraphNode;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.Serializable;
import java.util.*;

/**ComputationGraph network (neural network with arbitrary connection structure)
 */
public class ComputationGraph implements Serializable, Model {

    protected ComputationGraphConfiguration configuration;
    protected transient Solver solver;	//Used to call optimizers during backprop

    protected GraphVertex[] vertices;
    protected int[] topologicalOrder;

    private int numInputArrays;
    private int numOutputArrays;
    private int layerCount;

    private INDArray[] inputs;
    private INDArray[] labels;

    private NeuralNetConfiguration defaultConfiguration;

    public ComputationGraph(ComputationGraphConfiguration configuration){
        this.configuration = configuration;
        this.numInputArrays = configuration.getNetworkInputs().size();
        this.numOutputArrays = configuration.getNetworkOutputs().size();
        this.layerCount = configuration.getLayers().size();
        this.inputs = new INDArray[numInputArrays];
        this.labels = new INDArray[numOutputArrays];
        this.defaultConfiguration = configuration.getLayers().get(configuration.getLayers().keySet().iterator().next());    //TODO
    }

    /** The number of inputs to this network */
    public int getNumInputArrays(){
        return numInputArrays;
    }

    /** The number of output (arrays) for this network */
    public int getNumOutputArrays(){
        return numOutputArrays;
    }

    public void setInput(int inputNum, INDArray input){
        inputs[inputNum] = input;
    }

    public void setLabel(int labelNum, INDArray label){
        labels[labelNum] = label;
    }

    /** Initialize the ComputationGraph network */
    public void init(){
        //Initialization: create the GraphVertex objects, based on configuration structure

        Map<String,Layer> layerMap = new HashMap<>();
        for( Map.Entry<String,NeuralNetConfiguration> entry : configuration.getLayers().entrySet() ){
            String layerName = entry.getKey();
            NeuralNetConfiguration layerConf = entry.getValue();

            Layer layer = LayerFactories.getFactory(layerConf).create(layerConf, null, -1); //TODO: indices
            layerMap.put(layerName, layer);
        }

        Map<String,GraphNode> nodeMap = new HashMap<>();

        List<String> networkInputNames = configuration.getNetworkInputs();

        //Inputs for each layer
        Map<String,String[]> layerInputs = configuration.getLayerInputs();

        //Inputs for each GraphNode
        Map<String,String[]> graphNodeInputs = configuration.getGraphNodeInputs();

        int nVertices = layerMap.size() + nodeMap.size() + networkInputNames.size();
        this.vertices = new GraphVertex[nVertices];

        //All names: inputs, layers and graph nodes (index to name map)
        Map<Integer,String> allNames = new HashMap<>();
        Map<String,Integer> allNamesReverse = new HashMap<>();

        int i=0;
        for( String name : networkInputNames){
            int[] outVertices = null;   //TODO
            GraphVertex gv = new GraphVertex(name,i,outVertices);
            allNames.put(i,name);
            allNamesReverse.put(name,i);
            vertices[i++] = gv;
        }

        for( Map.Entry<String,Layer> layerEntry : layerMap.entrySet() ){
            int[] inputIndices = null;  //TODO
            int[] outputIndices = null;
            Layer l = layerEntry.getValue();
            String name = layerEntry.getKey();
            GraphVertex gv = new GraphVertex(name,i,inputIndices,outputIndices,l);
            allNames.put(i,name);
            allNamesReverse.put(name,i);
            vertices[i++] = gv;
        }

        for( Map.Entry<String,GraphNode> nodeEntry : nodeMap.entrySet() ){
            int[] inputIndices = null;  //TODO
            int[] outputIndices = null;
            GraphNode n = nodeEntry.getValue();
            String name = nodeEntry.getKey();
            GraphVertex gv = new GraphVertex(name,i,inputIndices,outputIndices,n);
            allNames.put(i,name);
            allNamesReverse.put(name,i);
            vertices[i++] = gv;
        }

        //Now: do another pass to set the input and output indices...
        //To get output indices: need to essentially build the graph in reverse...
        Map<String,List<String>> verticesOutputTo = new HashMap<>();    //Key: vertex. Values: vertices that this node is an input for


        for( GraphVertex gv : vertices ){
            String vertexName = gv.getVertexName();
            String[] vertexInputNames;

            if(gv.getLayer() != null){
                //vertex with layer
                vertexInputNames = layerInputs.get(vertexName);
            } else if(gv.getGraphNode() != null){
                //Vertex with node
                vertexInputNames = graphNodeInputs.get(vertexName);

            } else {
                //Input vertex
                vertexInputNames = null;
            }

            if(vertexInputNames == null) continue;

            int[] indices = new int[vertexInputNames.length];
            for( int j=0; j<vertexInputNames.length; j++ ){
                indices[j] = allNamesReverse.get(vertexInputNames[j]);
            }

            gv.setInputIndices(indices);

            //Build reverse network structure:
            for(String s : vertexInputNames){
                List<String> list = verticesOutputTo.get(s);
                if(list == null){
                    list = new ArrayList<>();
                    verticesOutputTo.put(s,list);
                }
                list.add(vertexName);   //Edge: s -> vertexName
            }
        }

        for( GraphVertex gv : vertices ) {
            String vertexName = gv.getVertexName();

            List<String> thisVertexOutputsTo = verticesOutputTo.get(vertexName);

            if(thisVertexOutputsTo == null) continue;   //Output vertex
            int[] indices = new int[thisVertexOutputsTo.size()];
            int j=0;
            for( String s : thisVertexOutputsTo ){
                indices[j++] = allNamesReverse.get(s);
            }
            gv.setOutputIndices(indices);
        }

        //At this point: each GraphVertex has the local connection structure, both for inputs and outputs
        for(GraphVertex gv : vertices ){
            System.out.println(gv);
        }



        //Given the graph structure, do a topological sort to define forward pass and flattening order:
        topologicalOrder = topologicalSortOrder();
    }

    /** Pretrain network with a single input and single output */
    public void pretrain(DataSetIterator iter){
        if(numInputArrays != 1 || numOutputArrays != 1) throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
            + " multiple inputs or outputs using a DataSetIterator");

        throw new UnsupportedOperationException("Not implemnted");
    }

    /** Pretrain network with multiple inputs and/or outputs */
    public void pretrain(Object multipleInputOutputIterator){
        throw new UnsupportedOperationException("Not implemnted");
    }

    public void fit(DataSet dataSet){
        if(numInputArrays != 1 || numOutputArrays != 1) throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
                + " multiple inputs or outputs using a DataSet");

        throw new UnsupportedOperationException("Not implemnted");
    }

    public void fit(DataSetIterator dataSetIterator){
        if(numInputArrays != 1 || numOutputArrays != 1) throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
                + " multiple inputs or outputs using a DataSetIterator");

        if(configuration.isPretrain()){

            throw new UnsupportedOperationException("Not implemented");
        }

        if(configuration.isBackprop()){
            while(dataSetIterator.hasNext()){
                DataSet next = dataSetIterator.next();
                if (next.getFeatureMatrix() == null || next.getLabels() == null)
                    break;

                boolean hasMaskArrays = next.hasMaskArrays();
                if(hasMaskArrays){
                    throw new UnsupportedOperationException("Not yet implemented");
//                    setLayerMaskArrays(next.getFeaturesMaskArray(), next.getLabelsMaskArray());
                }

                if(configuration.getBackpropType() == BackpropType.TruncatedBPTT) {
                    throw new UnsupportedOperationException("Not yet implemented");
//                    doTruncatedBPTT(next.getFeatureMatrix(),next.getLabels());
                }
                else {
                    setInput(0,next.getFeatureMatrix());
                    setLabel(0,next.getLabels());
                    if( solver == null ){
                        solver = new Solver.Builder()
                                .configure(defaultConfiguration)    //TODO; don't like this
                                .model(this).build();
                    }
                    solver.optimize();
                }

                if(hasMaskArrays){
                    throw new UnsupportedOperationException();
                    //clearLayerMaskArrays();
                }
            }
        }
    }

    public void fit(Object multipleInputOutputIterator){

        throw new UnsupportedOperationException("Not implemented");
    }

    public void fit(INDArray[] inputs, INDArray[] labels ){
//        setInputs(inputs);
//        setLabels(labels);


        if(configuration.isPretrain()){

            throw new UnsupportedOperationException("Not implemented");
        }

        if(configuration.isBackprop()){
            if(configuration.getBackpropType() == BackpropType.TruncatedBPTT){

                throw new UnsupportedOperationException("Not implemented");
            } else {
                if( solver == null) {
                    solver = new Solver.Builder()
                            .configure(conf())
//                            .listeners(getListeners())
                            .model(this).build();
                }

                solver.optimize();
            }
        }
    }

    /** Calculate a topological sort order for the vertices in the graph.
     * Note that this is used for
     * (a) working out what order to do forward pass,
     * (b) what order to do backprop (i.e., reverse of this)
     * (c) order to flatten parameters (and gradients)
     *  */
    public int[] topologicalSortOrder(){
        //https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
        int[] out = new int[vertices.length];
        int outCounter = 0;

        boolean[] processed = new boolean[vertices.length];

        //First: represent the graph more usefully as a Map<Integer,Set<Integer>>, where map represents edges i -> j
        // key represents j, set is set of i (inputs) for vertices j
        Map<Integer,Set<Integer>> inputEdges = new HashMap<>();
        for(GraphVertex gv : vertices){
            int[] vertexInputsFrom = gv.getInputVertexIndices();
            if(vertexInputsFrom == null || vertexInputsFrom.length == 0){
                inputEdges.put(gv.getIndex(),null);
                continue;
            }
            Set<Integer> set = new HashSet<>();
            for( int i : vertexInputsFrom ){
                set.add(i);
            }
            inputEdges.put(gv.getVertexIndex(),set);
        }

        //Now: do topological sort
        //Set of all nodes with no incoming edges: (this would be: input vertices)
//        LinkedList<GraphVertex> noIncomingEdges = new LinkedList<>();
        LinkedList<Integer> noIncomingEdges = new LinkedList<>();
        for( Map.Entry<Integer,Set<Integer>> entry : inputEdges.entrySet() ) {
            Set<Integer> inputsFrom = entry.getValue();
            if(inputsFrom == null || inputsFrom.size() == 0) {
                noIncomingEdges.add(entry.getKey());
            }
        }

//        for( GraphVertex v : vertices){
//            if(v.getNumInputArrays() == 0){
//                noIncomingEdges.add(v);
//                processed[v.getIndex()] = true;
//            }
//        }

        while(noIncomingEdges.size() > 0) {
            int next = noIncomingEdges.removeFirst();
            out[outCounter++] = next;   //Add to sorted list

            int[] vertexOutputsTo = vertices[next].getOutputVertexIndices();  //Edges: next -> vertexOutpusTo[...]
            //Remove edges next -> vertexOuputsTo[...] from graph;
            if(vertexOutputsTo != null ) {
                for (int i : vertexOutputsTo) {
                    Set<Integer> set = inputEdges.get(i);
                    set.remove(next);
                    if (set.size() == 0) {
                        noIncomingEdges.add(i); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }

        System.out.println("Topological sort order:");
        System.out.println(Arrays.toString(out));

        return out;
    }


    @Override
    public void computeGradientAndScore() {
        //Calculate activations (which are stored in each layer, and used in backprop)
        if(configuration.getBackpropType() == BackpropType.TruncatedBPTT) {
//            rnnActivateUsingStoredState(getInput(), true, true);
//            truncatedBPTTGradient();
            throw new UnsupportedOperationException();
        }
        else {
            feedForward(true);
            backprop();
        }
//        score = ((BaseOutputLayer<?>)getOutputLayer()).computeScore(calcL1(),calcL2(), true);
        throw new UnsupportedOperationException("Score calculation not implemented");
    }

    public Map<String,INDArray> feedForward(boolean train){

        //Do forward pass according to the topological ordering of the network
        for( int i=0; i<topologicalOrder.length; i++ ){
            GraphVertex current = vertices[topologicalOrder[i]];
            if(current.isInputVertex()){
                int[] outIndices = current.getOutputVertexIndices();

                for( int v : outIndices ){

                }

            } else {

            }
        }

    }

    protected void backprop(){


    }

    @Override
    public ComputationGraph clone(){

        throw new UnsupportedOperationException("Not implemented");
    }

    //------------------------------------------------------
    //Model methods:

    @Override
    public void fit() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public double score() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public INDArray params() {
        List<INDArray> list = new ArrayList<>(layerCount);
        for( int i=0; i<topologicalOrder.length; i++ ){
            if(!vertices[i].hasLayer()) continue;

            Layer l = vertices[i].getLayer();
            list.add(l.params());
        }

        return Nd4j.toFlattened('f',list);
    }

    @Override
    public int numParams() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public int numParams(boolean backwards) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setParams(INDArray params) {
        int idx = 0;
        for( int i=0; i<topologicalOrder.length; i++ ){
            if(!vertices[i].hasLayer()) continue;

            Layer layer = vertices[i].getLayer();
            int range = (layer instanceof BasePretrainNetwork ?
                    ((BasePretrainNetwork<?>)layer).numParamsBackprop() : layer.numParams());
            INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
            layer.setParams(get);
            idx += range;
        }
    }

    @Override
    public void applyLearningRateScoreDecay() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void fit(INDArray data) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public int batchSize() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public NeuralNetConfiguration conf() {
        return defaultConfiguration;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public INDArray input() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void validateInput() {
        //TODO
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public Map<String, INDArray> paramTable() {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void setParam(String key, INDArray val) {
        throw new UnsupportedOperationException("Not implemnted");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("Not implemnted");
    }


}
