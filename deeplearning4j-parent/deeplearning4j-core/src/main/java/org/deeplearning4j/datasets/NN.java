package org.deeplearning4j.datasets;


public class NN {

	int numnodes;
	int maxedges;
	float[] currentState; // current state of a node
	float[] nextState; // state at next timestep
	int[][] edges; // indices of partner nodes
	float[][] weights; // weights on edges
	
	// TODO: Extend to multi input/output
	int[] inputNodes = null; // indices of the input nodes
	float[] inputValues; 
	int[] outputNodes = null; // indices of the output nodes
	float[] outputValues; 	
	boolean firstUpdate = true;
	
	public NN(int nodes, int maxedges){
		numnodes = nodes;
		this.maxedges = maxedges;
		currentState = new float[nodes];
		nextState = new float[nodes];
		edges = new int[nodes][maxedges];
		weights = new float[nodes][maxedges];
	}
	
	public void init() {
		// set random number connections between nodes
		int numedges;
		int node;
		for (int i = 0; i < numnodes; i++){
			// set them all to index -1 (no partner)
			for (int j = 0; j < maxedges; j++){
				edges[i][j] = -1;
			}
			// now assign a random number of edges
			numedges = (int)(Math.random()*(maxedges + 1));
			for (int j = 0; j < numedges; j++){
				// find random node to bind to
				node = getRandomPartner(i, false);
				edges[i][j] = node;
				weights[i][j] = (float) Math.random();
			}
		}
	}
	
	/** 
	 * Input values into  the system via the input nodes
	 * 
	 * @param inputNodes indices of input nodes
	 * @param values activation value at each input node
	 */
	
	public void setInput(int[] inputNodes, float[] inputValues){
		this.inputNodes = inputNodes;
		this.inputValues = inputValues;
		applyInput();
		firstUpdate = true;
	}
	
	public void setOutput(int[] outputNodes, float[] outputValues){
		this.outputNodes = outputNodes;
		this.outputValues = outputValues;
		applyOutput();
		firstUpdate = true;
	}	
	
	private void applyInput(){
		for (int i = 0; i < inputNodes.length; i++){
			currentState[inputNodes[i]] = inputValues[i];
		}	
	}
	
	private void applyOutput(){
		for (int i = 0; i < outputNodes.length; i++){
			currentState[outputNodes[i]] = outputValues[i];
		}	
	}	
	
	public void update(){
		
		// Logic to show input applied to input nodes but to avoid
		// doubling up on first input
		if (firstUpdate){
			firstUpdate = false;
		} else {
			if (inputNodes != null){
				applyInput();
			}
			if (outputNodes != null){
				applyOutput();
			}			
		}
		
		// Firing parameters
		double threashold = 0.9;
		double theta = 1.0; // smaller theta makes the network more sensitive
		
		// walk through each node with a non-zero state
		for (int i = 0; i < numnodes; i++){
			// if > threashold, activate transmit signal along edges
			double x = currentState[i];
			double sigmoid = 1.0 / (Math.pow(Math.E, -x)) - theta;
			boolean active = sigmoid > threashold ? true : false;		
			if (active){
				// apply the state to the connected edges via the weights
				for (int j = 0; j < maxedges; j++){
					if (edges[i][j] != -1){
						nextState[edges[i][j]] += weights[i][j] * currentState[i];
					} else {
						break;
					}
				}
			}
		}
		// replace the current state with the new state
		for (int i = 0; i < numnodes; i++){
			currentState[i] = nextState[i];
			nextState[i] = 0f;
		}
	}
	
	public void reset(){
		for (int i = 0; i < numnodes; i++){
			currentState[i] = 0f;
		}
	}
	
	public float[] getState(){
		return currentState;
	}
	
	// TODO: Extend to multi output
	public float[] readOutput() {
		if (outputNodes == null){
			return null;
		} else {
			float[] values = new float[outputNodes.length];
			for (int i = 0; i < values.length; i++){
				values[i] = currentState[outputNodes[i]];
			}
			return values;
		}
	}
	
	public float[] readInput() { // This is called continuously from the GUI update event -- BAD
		float[] values = null;
		if (inputNodes != null){
			values = new float[inputNodes.length];
			for (int i = 0; i < values.length; i++){
				values[i] = currentState[inputNodes[i]];
			}
		}
		return values;
	}	
	
	private int getRandomPartner(int self, boolean selfAllowed){
		int node = (int)(Math.random()*numnodes);
		if (!selfAllowed){
			while (node == self){
				node = (int)(Math.random()*numnodes);
			}
		}
		return node;	
	}
	
	public String toString(){
		String s = "Network statistics:\n";
		s += "Total nodes: "+numnodes+"\n";
			
		// connectivity stats
		int numedges = 0;
		for (int i = 0; i < numnodes; i++){
			for (int j = 0; j < maxedges; j++){
				if (edges[i][j] != -1){
					numedges++;
				} else {
					break;
				}
			}
			s += "";
		}
		s += "Total edges: "+numedges+"\n";
		s += "Edge density: "+(float)numedges/(float)numnodes+"\n";
		s += "Occupancy: "+(float)numedges/(float)(maxedges*numnodes)+"\n";
		
		return s;
	}
	
	public String getConnections(){
		String s = "Connections:\n";
		for (int i = 0; i < numnodes; i++){
			s += i+": ";
			for (int j = 0; j < maxedges; j++){
				if (edges[i][j] != -1){
					s += edges[i][j] + ",";
				}
			}
			s += "\n";
		}
		
		return s;		
	}
	
	public String getWeights(){
		String s = "Weights:\n";
		for (int i = 0; i < numnodes; i++){
			s += i+": ";
			for (int j = 0; j < maxedges; j++){
				if (edges[i][j] != -1){
					s += weights[i][j] + ",";
				}
			}
			s += "\n";
		}
		
		return s;		
	}	
	
	public String getCurrentState(){
		String s = "Current State:\n";
		for (int i = 0; i < numnodes; i++){
			s += i+": ";
			s += currentState[i] + "\n";	
		}
		
		return s;		
	}	
	
	public String getNextState(){
		String s = "Next State:\n";
		for (int i = 0; i < numnodes; i++){
			s += i+": ";
			s += nextState[i] + "\n";	
		}
		
		return s;		
	}		
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// For command line learning
		NN net = new NN(10,5);
		net.init();
		net.setInput(new int []{0,1,2,3,4}, new float[] {1.0f, 0.5f, 0.2f, 0.9f, 0.7f});
		System.out.println(net.getConnections());
		System.out.println(net.getWeights());
		while (true){
			System.out.println(net.getCurrentState());
			net.update();	
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//System.out.println(net.getNextState());
		//System.out.print(net.toString());
	}

}
