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

package org.datavec.cli.records.reader;


import org.datavec.api.conf.Configuration;
import org.datavec.api.writable.Text;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Writable;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

/**
 * Hardcoded for serial only access (for now)
 * 
 * @author josh
 *
 */
public class LineRecordReader extends FileRecordReader  {
    //private AbstractTfidfVectorizer tfidfVectorizer;
    private Collection<Collection<Writable>> records = new ArrayList<>();
    //private Iterator<Collection<Writable>> recordIter;
    private BufferedReader textFileBufferedReader = null;
    private Scanner textFileScanner = null;
    private Configuration conf;
    
    private String currentLine = "";
    private String currentPath = "";
    //private String currentRelativePath = "";
    //private String nextLine = "";
    //private String nextPath = "";


    @Override
    public void initialize(InputSplit split) throws IOException, InterruptedException {
        initialize(new Configuration(),split);
    }

    /**
     * Need to look at the lines in a set of files in directories
     * 
     */
    @Override
    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf,split);
//        tfidfVectorizer = new AbstractTfidfVectorizer();
//        tfidfVectorizer.initialize(conf);
//        tfidfVectorizer.fit(this, new Vectorizer.RecordCallBack() {
/*            @Override
            public void onRecord(Collection<Writable> record) {
                records.add(record);
            }
        });
*/
        //recordIter = records.iterator();
        //System.out.println( "datavec-LineRR:init() " + recordIter.hasNext() );
        
        /*
    	try {
			this.textFileBufferedReader = new BufferedReader( new FileReader( this.iter.next().getAbsolutePath() ) );
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}  
    	
    	this.nextLine = this.textFileBufferedReader.readLine();
    	this.currentPath = 
        */
        
		this.currentFile = this.iter.next();
		this.currentPath = this.currentFile.getAbsolutePath();
		//this.textFileBufferedReader = new BufferedReader( new FileReader( this.currentPath ) );
        this.textFileScanner = new Scanner( new FileInputStream( this.currentPath ) );
    }
    
    /*
    private void readNextLine() {

    }
    */
    
    private void rotateScannerToNextFile() throws FileNotFoundException {
    	
	//	System.out.println("> rotate reader ");
		this.currentFile = this.iter.next();
		this.currentPath = this.currentFile.getAbsolutePath();
		//this.textFileBufferedReader = new BufferedReader( new FileReader( this.currentPath ) );
    	
		this.textFileScanner = new Scanner( new FileInputStream( this.currentPath ) );
    	
    }
    

    /**
     * Major difference here: we ALWAYS append label as string
     * 
     * 	-	we've also kicked the responsibility of indexing labels UP a level to the vectorization engine for text
     * 
     * 
     * 
     * NEED to look at the file iterator (iter)
     * 
     */
    @Override
    public Collection<Writable> next() {
    	
    	//String currentLine = this.nextLine;
    	//String returnedLabelPath = this.currentPath;
    	
    	this.currentLine = null;  
    	boolean noMoreFiles = false;
    	
    	
    	/*
    	if ( null == this.textFileBufferedReader ) {
	    	
	    	
	    	
	    	try {

    	    	try {
    	    		System.out.println("> init reader ");
    	    		this.currentFile = this.iter.next();
    	    		this.currentPath = this.currentFile.getAbsolutePath();
    				this.textFileBufferedReader = new BufferedReader( new FileReader( this.currentPath ) );
    			} catch (FileNotFoundException e1) {
    				// TODO Auto-generated catch block
    				e1.printStackTrace();
    			}  
	    		
	    		
	    		// now setup the next line pull so the .hasNext() method returns correctly
	    		while ( ( this.currentLine = this.textFileBufferedReader.readLine() ) == null ) {
	    			
	    			System.out.println("> looking for new reader 1");
	    			
	    	    	try {
	    	    		this.currentFile = this.iter.next();
	    	    		this.currentPath = this.currentFile.getAbsolutePath();
	    				this.textFileBufferedReader = new BufferedReader( new FileReader( this.currentPath ) );
	    			} catch (FileNotFoundException e1) {
	    				// TODO Auto-generated catch block
	    				e1.printStackTrace();
	    			}  
	    			
	    		}
	
	    	} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	    	
    	} else {
    		
    		try {
				while ( ( this.currentLine = this.textFileBufferedReader.readLine() ) == null ) {
					
					System.out.println("> looking for new reader 2");
					
					try {
						this.currentFile = this.iter.next();
						this.currentPath = this.currentFile.getAbsolutePath();
						this.textFileBufferedReader = new BufferedReader( new FileReader( this.currentPath ) );
					
					} catch (NoSuchElementException noMoreFilesEx) {
					
						noMoreFiles = true;
						break;
					
					} catch (FileNotFoundException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}  
					
				}
				
				if (!noMoreFiles) {
					System.out.println( "Reader: Found file: " + this.currentPath );
				} else {
					
				}
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    		
    		
    	}
    	*/
    	
    	while (!this.textFileScanner.hasNextLine()) {
    		
    		try {

    			this.rotateScannerToNextFile();
    			
			} catch (FileNotFoundException e) {

				//e.printStackTrace();
				noMoreFiles = true;
			}
    		
    	}
    	
    	
    	
    	if (noMoreFiles) {

    		//System.out.println("> ran out of files!");
    		this.iter = null;
    		this.currentFile = null;
    		//this.textFileBufferedReader = null;
    		throw new NoSuchElementException("No more elements found!");
    		
    	} else {
    		
    		this.currentLine = this.textFileScanner.nextLine();
    		
    	}
    	
    	/*
    	// if the line is null and we dont have any more files to process, then we're done
    	if ( null == this.currentLine && this.textFileBufferedReader == null ) {
    		this.iter = null;
    		this.currentFile = null;
    		this.textFileBufferedReader = null;

    		System.out.println("> ran out of readers!");
    		throw new NoSuchElementException("No more elements found!");
    		    		
    	}
    	*/
    	    	    	
        Collection<Writable> record = new ArrayList<>();
        
        record.add(new Text( this.currentLine ));
        record.add(new Text( this.getCurrentDirectoryLabelPath() ));

        return record;
        
    	
    }

    @Override
    public boolean hasNext() {
		return null != this.textFileScanner &&
				this.textFileScanner.hasNextLine() ||
				iter != null && iter.hasNext();

	}
    
    public String getCurrentDirectoryLabelPath() {
        //return labels.indexOf(currentFile.getParentFile().getName());
    	return currentFile.getParentFile().getName();
    }
    

    @Override
    public void close() throws IOException {

    }

    @Override
    public void setConf(Configuration conf) {
        this.conf = conf;
    }

    @Override
    public Configuration getConf() {
        return conf;
    }


}
