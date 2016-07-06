package org.canova.cli.vectorization;

import java.io.IOException;
import java.util.Collection;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.writable.Writable;
import org.canova.cli.shuffle.Shuffler;
import org.canova.cli.subcommands.Vectorize;
import org.canova.cli.transforms.image.NormalizeTransform;

import com.google.common.base.Strings;

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
