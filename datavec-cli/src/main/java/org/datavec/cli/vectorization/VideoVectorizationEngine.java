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
