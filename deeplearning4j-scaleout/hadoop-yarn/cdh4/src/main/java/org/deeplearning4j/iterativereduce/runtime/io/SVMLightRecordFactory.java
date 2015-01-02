package org.deeplearning4j.iterativereduce.runtime.io;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.FeatureUtil;


public class SVMLightRecordFactory {


//	  public static final int FEATURES = 10000;
	  //ConstantValueEncoder encoder = null;
	  private boolean useBiasTerm = false;
	  private int featureVectorSize = 1;
	  //private String schema = "";
	  private int inputValues = 0;
	  //private int outputValues = 0;
//	  double[] output_zeros; // = new double[this.outputValues];
//	  double[] input_zeros; 
	  
	  /**
	   * dont need to parse the schema, just trust that the incoming vectors are the right size
	   * 
	   */
	  public SVMLightRecordFactory(int featureVectorSize) {
	    
		  this.featureVectorSize = featureVectorSize;
//		  this.schema = schema;
	    //this.encoder = new ConstantValueEncoder("body_values");
//	    this.parseSchema();
	  }
	  
	  public void setUseBiasTerm() {
		  this.useBiasTerm = true;
	  }
	  /*
	  public void parseSchema() {
		  
		  String[] parts = this.schema.split("\\|");
		  
		  //System.out.println("parts " + parts.length);
		  
		  for ( int x = 0; x < parts.length; x++) {
			  
			  //System.out.println("part " + x + ": " + parts[x]);

			  // now split schema into in/out parts
			  String[] type_vals = parts[x].trim().split(":");
			  if (type_vals[0].equals("i")) {
				  
				  // input vals
				  this.inputValues = Integer.parseInt(type_vals[1]);
//				  this.input_zeros = new double[this.inputValues];
				  this.featureVectorSize = this.inputValues;
				  
			  } else if (type_vals[0].equals("o")) {
				  
				  // output vals
				  this.outputValues = Integer.parseInt(type_vals[1]);
	//			  this.output_zeros = new double[this.outputValues];
				  
			  }
			  
		  }
		  
	  }
	  */
//	  @Override
	  public int getInputVectorSize() {
		  return this.inputValues;
	  }
	  
//	  @Override
/*	  public int getOutputVectorSize() {
		  return this.outputValues;
	  }
*/	  

	  
	  // doesnt really do anything in a 2 class dataset
/*	  @Override
	  public String GetClassnameByID(int id) {
	    return String.valueOf(id); // this.newsGroups.values().get(id);
	  }
*/	  
	 /* 
	  public void clearVector(Vector v) {
		  
		  Iterator<Element> it = v.iterateNonZero();
		  while (it.hasNext()) {
		    Element e = it.next();
		    e.set(0);
		  }
		  
	  }
	  */
	  // INDArray vector = Nd4j.create(10);
	  
	  /**
	   * 
	   * example line: "-1 1:0.43 3:0.12 9284:0.2 # abcdef"
	   * 
	   * @param line
	   * @param input_vec
	   * @param output_vec
	 * @throws Exception 
	   */
	  public void parseFromLine(String line, INDArray input_vec, INDArray output_vec) {
	    
	    //String[] inputs_outputs = line.split("\\|");
		  // remove comments
		  String[] vector_and_comments = line.split("#");

		  // now use only the left hand part
		  String[] inputs_outputs = vector_and_comments[0].split(" ");
	    
	    //System.out.println("in: " + inputs_outputs[0]);
	    //System.out.println("out: " + inputs_outputs[1]);
	    
	    String label = inputs_outputs[0].trim();
	    //String[] outputs = inputs_outputs[1].trim().split(" ");
	    
	    
	    // clear it
	    input_vec.muli(0.0); // clear vector
	    
	    int startFeatureIndex = 0;
	    // dont know what to do the the "namespace" "f"
	    if (this.useBiasTerm) {
	    	// input_vec.set(0, 1.0);
	    	input_vec.putScalar(0, 1.0);
	    	startFeatureIndex = 1;
	    }
	    
	    for (int x = 1; x < inputs_outputs.length; x++) {
	      
	    	//System.out.println("> DEbug > part: " + parts[x]);
	    	
	      String[] feature = inputs_outputs[x].split(":");
	      
	      if ("#".equals( feature[0].trim() ) ) {
	    	  
	    	  // comment
	    	  
	      } else {
		      // get (offset) feature index and hash as necesary
		      int index = (Integer.parseInt(feature[0]) + startFeatureIndex); // % this.featureVectorSize;
		      
		      double val = Double.parseDouble(feature[1]);
		      	      
		      if (index < this.featureVectorSize) {
		    	  
		    	  input_vec.putScalar(index, val);
		    	  
		      } else {
		        
		    	  // Should we throw an exception here?
		        System.err.println("Could Hash: " + index + " to " + (index % this.featureVectorSize));
		        
		      }
		      
	      }
	      
	    }
	    
	    //FeatureUtil.toOutcomeVector(index, numOutcomes)
	    
	    output_vec.muli(0.0); // clear vector
	    
//	      boolean noLabelException = false;
	      double val = 0;
	      val = Double.parseDouble( label );
	      output_vec.putScalar(0, val);


	    
	  }

	  public int getFeatureVectorSize() {
		return this.featureVectorSize;
	}	
	
}
