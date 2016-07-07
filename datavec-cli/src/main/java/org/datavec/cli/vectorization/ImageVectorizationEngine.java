/*
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.cli.vectorization;

import java.io.IOException;
import java.util.Collection;

import org.datavec.api.writable.Writable;
import org.datavec.cli.transforms.image.NormalizeTransform;
import org.datavec.cli.shuffle.Shuffler;

/**
 * Reads from InputFormats where (generally, but up to InputFormat) each Writable in Collection is a pixel
 * 
 * Writes back out to the OutputFomat where we are assuming the last element is the double representing the class index
 * 
 * @author josh
 *
 */
public class ImageVectorizationEngine extends VectorizationEngine {

	/**
	 * In this case we are assuming that the Image input format gave us basically raw pixels
	 * 
	 * 
	 * Thoughts
	 * 		-	Inside the vectorization engine is a great place to put a pluggable transformation system [ TODO: v2 ]
	 * 			-	example: MNIST binarization could be a pluggable transform
	 * 			-	example: custom thresholding on blocks of pixels
	 * 
	 * 
	 */
	@Override
	public void execute() throws IOException {

		NormalizeTransform normalizer = new NormalizeTransform();
		
		// 1. collect stats for normalize
        while (reader.hasNext()) {
            
        	// get the record from the input format
        	Collection<Writable> w = reader.next();
        	normalizer.collectStatistics(w);

        }
        
		// 2. reset reader
        
        reader.close();
        //RecordReader reader = null;
		try {
			this.reader = inputFormat.createReader(split, conf);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// 3. transform data
		
	      if (shuffleOn) {
	    	  
	    	  Shuffler shuffle = new Shuffler();
			
			
			//int x = 0;
	        while (reader.hasNext()) {
	            
	        	// get the record from the input format
	        	Collection<Writable> w = reader.next();
						if (normalizeData) {
              normalizer.transform(w);
            }

						// the reader did the work for us here
	        	//writer.write(w);
	        	shuffle.addRecord(w);
	 
	        }
	        
	        // now write the shuffled data out
	        
			while (shuffle.hasNext()) {
				
				Collection<Writable> shuffledRecord = shuffle.next();
				writer.write( shuffledRecord );
				
			}	  
	        

	      } else {
	    	  

		        while (reader.hasNext()) {
		            
		        	// get the record from the input format
		        	Collection<Writable> w = reader.next();
							if (normalizeData) {
                normalizer.transform(w);
              }

							// the reader did the work for us here
		        	writer.write(w);
		 
		        }

	    	  
	      }
	      
        reader.close();
        writer.close();		
		
	//	System.out.println( "ImageVectorizationEngine > execute() [ END ]" );
        
		
	}

	
}
