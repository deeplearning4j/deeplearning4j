package org.canova.cli.vectorization;

import java.io.IOException;
import java.util.Collection;

import org.etl4j.api.writable.Writable;

public class AudioVectorizationEngine extends VectorizationEngine {

	/**
	 * 
	 * 
	 */
	@Override
	public void execute() throws IOException {

		System.out.println( "AudioVectorizationEngine > execute() [ START ]" );
		
		
        while (reader.hasNext()) {
            
        	// get the record from the input format
        	Collection<Writable> w = reader.next();
        	
        	// the reader did the work for us here
        	writer.write(w);
         }
	

        reader.close();
        writer.close();		
		
		System.out.println( "AudioVectorizationEngine > execute() [ END ]" );
        
		
	}
}
