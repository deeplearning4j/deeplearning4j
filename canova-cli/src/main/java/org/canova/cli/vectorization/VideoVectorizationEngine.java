package org.canova.cli.vectorization;

import java.io.IOException;

public class VideoVectorizationEngine extends VectorizationEngine {

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

		System.out.println( "VideoVectorizationEngine > execute() [ START ]" );
		/*
		int x = 0;
        while (reader.hasNext()) {
            
        	// get the record from the input format
        	Collection<Writable> w = reader.next();
        	
        	// the reader did the work for us here
        	writer.write(w);
         }
	*/

        reader.close();
        writer.close();		
		
		System.out.println( "VideoVectorizationEngine > execute() [ END ]" );
        
		
	}
}
