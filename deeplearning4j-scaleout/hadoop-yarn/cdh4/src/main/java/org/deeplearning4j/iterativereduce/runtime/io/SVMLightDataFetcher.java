package org.deeplearning4j.iterativereduce.runtime.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

/**
 * 
 * Currently we only support non-negative labels
 *  - this is due to the issue where we have not yet accounted for the scenario
 *    where each split needs the same label conversion heuristic 
 *    we're just using the class labels directly as indexes
 *   
 *   
 * @author josh
 *
 */
public class SVMLightDataFetcher extends BaseDataFetcher implements DataSetFetcher {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	//private transient MnistManager man;
	//public final static int NUM_EXAMPLES = 60000;

	int currentVectorIndex = 0;
	
	Text tmpTextRecord = new Text("");
	TextRecordParser record_reader = null;
	SVMLightRecordFactory vector_factory = null;
	
	// tells the record factory how to layout the vectors in|out
	//private String vectorSchema = "i:784 | o:10";
	//int maxFeatureCount = 0;

	boolean bCacheIsHot = false;

	/**
	 * For now we'll just give the fetcher an instantiated parser
	 * - may can be a more elegant way to do this
	 * 
	 * We're assuming a text input format with a Metronome vector layout
	 * 
	 * @param hdfsLineParser
	 * @throws IOException
	 */
	public SVMLightDataFetcher( TextRecordParser hdfsLineParser, int featureCount, int numOutcomes ) throws IOException {

		this.record_reader = hdfsLineParser;
		//this.maxFeatureCount = featureCount;
		
		//numOutcomes = 10;
		this.cursor = 1;
		this.numOutcomes = numOutcomes;
		this.inputColumns = featureCount; //ArrayUtils.flatten(image).length;

		this.vector_factory = new SVMLightRecordFactory( this.inputColumns );
	}

	/**
	 * Converts a line of Metronome record format to the Pair<Image,Label> format expected by the dataset code
	 * 
	 * data comes into the this point already normalized by the conversion to text format
	 * 
	 * TODO:
	 * - look at efficiency here, prolly need to let the vector factory convert straight to Matrix recs
	 * 
	 * @param line
	 * @return
	 */
	public Pair<INDArray,INDArray> convertTextLineToInputPair( String line ) {
		
		//Vector v_in = new RandomAccessSparseVector( this.vector_factory.getFeatureVectorSize());
		//Vector v_out = new RandomAccessSparseVector( this.vector_factory.getOutputVectorSize());
		
		System.out.println("line: " + line );
		
		INDArray vec_in = Nd4j.create( this.inputColumns );
		INDArray vec_out = Nd4j.create( 1 );
				
		try {
			this.vector_factory.parseFromLine(line, vec_in, vec_out);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
//		Matrix input = new DenseMatrix( 1, v_in.size() );
//		input.viewRow(0).assign(v_in);
			
//		Matrix label = new DenseMatrix( 1, v_out.size() );
//		label.viewRow(0).assign(v_out);
	/*	
		boolean found = false;
		
		for (int col = 0; col < label.numCols(); col++) {
			
			if (label.get(0, col) > 0) {
				found = true;
				break;
			}
			
		}
		
		if (!found) {
			
			throw new IllegalStateException("Found a matrix without an outcome");
			
		}
*/		
		
		return new Pair<>(vec_in, vec_out);
		
	}

	/**
	 * NOTE:
	 * 
	 * - be sure to preserve the data normalization
	 * 
	 * TODO:
	 * - cache the read vectors into batches
	 * 		- do we cache the batch or just let that get recreated?
	 * 
	 */
	@Override
	public void fetch(int numExamples) {
		
//	    Text value = new Text();	    
	    boolean result = true;
		
		
		//if (!hasMore()) {
		if (false == this.record_reader.hasMoreRecords()) {
			
			throw new IllegalStateException("Unable to get more; there are no more images");
			
		}

		// so here we replace the MnistManager with the Hadoop based record reader to start pulling text based
		// lines off hdfs

        List<DataSet> vectorBatch = new ArrayList<>();
        

		// so  we need to fill up a batch
		// - if we cannot fill a batch, we need to get the tail end of the records
		// - we need to come up w some way to bound the max number of records a worker touches
		
		for (int i = 0; i < numExamples; i++, cursor++ ) {
			
			if (false == this.record_reader.hasMoreRecords()) {
				
				System.out.println( "early kickout of svmLight hdfs data fetcher" );
				break;
				
			}
			
			try {
				
				result = this.record_reader.next( this.tmpTextRecord );
				
				if (false == result) {
					System.err.println( "SVMLightDataFetcher > hit no recs " );
					break;
				}
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
			String valString = this.tmpTextRecord.toString().trim();
			
			if (valString.equals("")) {
				
				System.err.println( "SVMLightDataFetcher > hit blank line " );
				
			} else {
			
				//toConvert.add( this.convertMetronomeTextLineToMatrixInputPair( value.toString() ));
				Pair<INDArray, INDArray> tmpPair = this.convertTextLineToInputPair(valString);
				
				//DataSet tmpDS = new DataSet( tmpPair.getFirst(), tmpPair.getSecond() );
				
				//System.out.println( "feature columns: " + tmpDS.getFeatures().columns() );
				//System.out.println( "labels columns: " + tmpDS.getLabels().columns() );
				
				//tmpDS.getLabels().linearIndex( 1 );
				
				//INDArray arTmp = FeatureUtil.toOutcomeVector(tmpPair.getSecond().getInt(0),numOutcomes);
				
				//System.out.println( "outcomes vec: {" + arTmp.getDouble(0) + ", " + arTmp.getDouble(1) + "} " );
				
				
				vectorBatch.add( new DataSet( tmpPair.getFirst(), FeatureUtil.toOutcomeVector(tmpPair.getSecond().getInt(0),numOutcomes)));
				
			}
		}

		//System.out.println( "number vectors: " + vectorBatch.size() );

		initializeCurrFromList( vectorBatch );



	}

	@Override
	public void reset() {
		cursor = 1;
		this.record_reader.reset();
	}
	
	@Override
	public boolean hasMore() {
		//return cursor < totalExamples;
		return this.record_reader.hasMoreRecords();
	}
}
