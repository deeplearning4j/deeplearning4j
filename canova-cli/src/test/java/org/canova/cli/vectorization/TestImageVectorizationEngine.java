package org.canova.cli.vectorization;

public class TestImageVectorizationEngine {

	
	/*
	@Test
	public void testInputFormatWithImageEngine() throws IOException, InterruptedException {

        String TEMP_ROOT = System.getProperty("user.home");
        String MNIST_ROOT = TEMP_ROOT + File.separator + "MNIST" + File.separator;   
        
        String MNIST_Filename = MNIST_ROOT + MNISTRecordReader.TRAINING_FILES_FILENAME_UNZIPPED;
    	
    	// 1. check for the MNIST data first!
    	
    	// does it exist?
    	
    	// if not, then let's download it
    	
        System.out.println( "Checking to see if MNIST exists locally: " + MNIST_ROOT );
        
        if(!new File(MNIST_ROOT).exists()) {
        	System.out.println("Downloading and unzipping the MNIST dataset locally to: " + MNIST_ROOT );
            new MnistFetcher().downloadAndUntar();
        } else {
        	
        	System.out.println( "MNIST already exists locally..." );
        	
        }
        
        if ( new File(MNIST_Filename).exists() ) {
        	System.out.println( "The images file exists locally unzipped!" );
        } else {
        	System.out.println( "The images file DOES NOT exist locally unzipped!" );
        }		
		
        
        RecordReader reader = new MNISTRecordReader();
        
        //ClassPathResource res = new ClassPathResource( MNIST_Filename );
        File resMNIST = new File( MNIST_Filename );
        InputStream targetStream = new FileInputStream( resMNIST );
       // resMNIST.
        
        reader.initialize(new InputStreamInputSplit( targetStream, resMNIST.toURI() ) );
        
        assertTrue(reader.hasNext());
        
        
        File out = new File("/tmp/mnist_svmLight_out.txt");
        //out.deleteOnExit();
        RecordWriter writer = new SVMLightRecordWriter(out,true);
		
		ImageVectorizationEngine engine = new ImageVectorizationEngine();
//		engine.initialize( new InputStreamInputSplit( targetStream, resMNIST.toURI() ), reader, writer);
		
		// setup split and reader
		

        String datasetInputPath = "";
        
        engine.execute();

        // check out many records are in the output
		
	}
*/
}
