package org.canova.cli.records.reader;


import org.canova.api.conf.Configuration;
import org.canova.api.io.data.IntWritable;
import org.canova.api.io.data.Text;
import org.canova.api.records.reader.impl.FileRecordReader;
import org.canova.api.split.InputSplit;
import org.canova.api.vector.Vectorizer;
import org.canova.api.writable.Writable;
//import org.canova.nd4j.nlp.vectorizer.TfidfVectorizer;

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
    //private TfidfVectorizer tfidfVectorizer;
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
//        tfidfVectorizer = new TfidfVectorizer();
//        tfidfVectorizer.initialize(conf);
//        tfidfVectorizer.fit(this, new Vectorizer.RecordCallBack() {
/*            @Override
            public void onRecord(Collection<Writable> record) {
                records.add(record);
            }
        });
*/
        //recordIter = records.iterator();
        //System.out.println( "Canova-LineRR:init() " + recordIter.hasNext() );
        
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
