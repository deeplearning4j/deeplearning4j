package org.canova.cli.transforms.text.nlp;


import org.apache.commons.math3.util.Pair;
import org.apache.lucene.analysis.snowball.SnowballAnalyzer;
import org.canova.api.berkeley.Counter;
import org.canova.api.conf.Configuration;
import org.canova.api.io.data.DoubleWritable;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.vector.Vectorizer.RecordCallBack;
import org.canova.api.writable.Writable;
import org.canova.cli.transforms.Transform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import org.canova.nlp.metadata.DefaultVocabCache;
import org.canova.nlp.metadata.VocabCache;
import org.canova.nlp.stopwords.StopWords;
import org.canova.nlp.tokenization.tokenizer.TokenPreProcess;
import org.canova.nlp.tokenization.tokenizer.Tokenizer;
import org.canova.nlp.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.canova.nlp.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.canova.nlp.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * Design Notes
 * 		-	dropped inheritance hierarchy of previous impl, would not fit a parallel framework's scale needs
 * 		-	we dont yet know what the principle design factors are going to be, so shying away from putting more interfaces in place
 * 		-	need to focus on a two pass mentality:
 * 			1. collect stats about the dataset ( so we can calc TF-IDF factors here )
 * 			2. apply the stats to the source data in the transform pass
 * 
 * 			-	this will match up better to a MapReduce / Spark runtime and allow us to scale out in the v2 design
 * 
 * 		-	this is considered the v1 design
 * 			-	we're trying to get the basic serial mechanics down across { csv, text, image }
 * 			-	setup some design stuff for next design wave (v2)
 * 		-	v2 design ideas
 * 			-	want to add timeseries class stuff: { audio, video, general timeseries }
 * 		-	v3 design ideas
 * 			-	parallelization runtime support 
 * 
 * 
 * 
 * @author josh
 *
 */
public class TfidfTextVectorizerTransform implements Transform {

    protected TokenizerFactory tokenizerFactory;
    protected int minWordFrequency = 0;
    public final static String MIN_WORD_FREQUENCY = "org.nd4j.nlp.minwordfrequency";
    public final static String STOP_WORDS = "org.nd4j.nlp.stopwords";
    public final static String TOKENIZER = "org.canova.nlp.tokenizerfactory";
    protected Collection<String> stopWords;
    protected VocabCache cache;

	// we want to track the label counts to understand the class balance
	// layout: { columnName, columnID, occurenceCount }
	public Map<String, Pair<Integer, Integer>> recordLabels = new LinkedHashMap<>();
	
	final EndingPreProcessor preProcessor = new EndingPreProcessor();

	public int getVocabularySize() {
		return this.cache.vocabWords().size();
	}
	
	public void debugPrintVocabList() {
		
		System.out.println( "Vocabulary Words: " );
		
        for (int i = 0; i < cache.vocabWords().size(); i++) {

        	System.out.println( i + ". " + cache.wordAt(i) );
        }
		
		
	}

    public void doWithTokens(Tokenizer tokenizer) {
    	
    	//SnowballAnalyzer stemmer;
    	
        Set<String> seen = new HashSet<>();
        
        //System.out.println( "doWithTokens ... " );
        while(tokenizer.hasMoreTokens()) {
            
        	String token = tokenizer.nextToken();
            
        	//System.out.println( "> " + token );
            
        	cache.incrementCount(token);
            
        	//System.out.println( "> vocab count " + cache.vocabWords().size() );
        	
        	if(!seen.contains(token)) {
                cache.incrementDocCount(token);
            }
        }
        
    }


    public TokenizerFactory createTokenizerFactory(Configuration conf) {
        String clazz = conf.get(TOKENIZER,DefaultTokenizerFactory.class.getName());
        
        //System.out.println( "Class for tokenizer factory: " + clazz );
        
        try {
            Class<? extends TokenizerFactory> tokenizerFactoryClazz = (Class<? extends TokenizerFactory>) Class.forName(clazz);
            return tokenizerFactoryClazz.newInstance();
        } catch (Exception e) {
           throw new RuntimeException(e);
        }
    }	
    
    

    public void initialize(Configuration conf) {
        tokenizerFactory = createTokenizerFactory(conf);
        tokenizerFactory.setTokenPreProcessor(new TokenPreProcess() {
            @Override
            public String preProcess(String token) {
                //token = token.toLowerCase();
				if (!token.startsWith("http://")) {
                    token = token.replaceAll("[^a-zA-Z ]", "").toLowerCase();
                }
				String base = preProcessor.preProcess(token);
                base = base.replaceAll("\\d", "d");
                if (base.endsWith("ly") || base.endsWith("ing"))
                    System.out.println();
                return base;
            }
        });

        
        minWordFrequency = conf.getInt(MIN_WORD_FREQUENCY,5);
        stopWords = conf.getStringCollection(STOP_WORDS);
        if(stopWords == null || stopWords.isEmpty())
            stopWords = StopWords.getStopWords();
        cache = new DefaultVocabCache(minWordFrequency);

        
        
    }


    protected Counter<String> wordFrequenciesForSentence( String sentence ) {
    	
        Tokenizer tokenizer = tokenizerFactory.create( sentence );
        
        Counter<String> ret = new Counter<>();
        
        String token;
        
        while(tokenizer.hasMoreTokens()) {
        
        	try {
        	
        		token = tokenizer.nextToken();
        		
            	ret.incrementCount( token, 1.0 );

        	} catch (NoSuchElementException e) {
        		System.out.println( "Bad Token" );
        	}
        
        }
                
        return ret;
    }

    
    public INDArray convertTextRecordToTFIDFVector( String textRecord ) {
    	
    	// here sentence represents a full document
    	Counter<String> wordFrequenciesForDocument = this.wordFrequenciesForSentence( textRecord );
    	
        INDArray ret = Nd4j.create(cache.vocabWords().size());

		int totalDocsInCorpus = (int)cache.numDocs();

        // for each word in our known vocabulary ..
        for(int i = 0; i < cache.vocabWords().size(); i++) {
        	
        	String term = cache.wordAt( i );
        	
        	// [TF] get the count of the word for this document (parameter for TF)
            int termFreq_ForThisSpecificDoc = (int) wordFrequenciesForDocument.getCount( cache.wordAt( i ) );
            
            // [IDF]
            // oddly enough, the vocabCache just pulls the doc frequency for a term
    		int numberOfDocsThisTermAppearsIn = (int) cache.idf(term);
            
            
    		double tf_term = NLPUtils.tf( termFreq_ForThisSpecificDoc );
    		double idf_term = NLPUtils.idf( totalDocsInCorpus, numberOfDocsThisTermAppearsIn );
    		
    		double tfidf_term = NLPUtils.tfidf( tf_term, idf_term );
            
            ret.putScalar( i, tfidf_term );
            
        }
                
        return ret;
    	
    }

	/**
	 * Collect stats from the raw record (first pass)
	 * 
	 * Schema:
	 * 
	 * 		Writable[0]: go dogs, go 1
	 * 		Writable[1]: label_A
	 * 
	 * 1. take the raw record, split off the label and cache it
	 * 2. take the remaining part of the raw record and run it through the stat collector for tf-idf
	 * 		-	previously this was the "fit()" method
	 *  
	 **/ 
	@Override
	public void collectStatistics(Collection<Writable> vector) {

		String label;
		String sentence;
		
		
		// 1. extract label and sentence from vector
		
		label = vector.toArray()[1].toString();
		
		sentence = vector.toArray()[0].toString();
		
		this.trackLabel(label);
		
	//	System.out.println( "sentence: " + sentence );
		
		
            Tokenizer tokenizer = tokenizerFactory.create( sentence );
        
            cache.incrementNumDocs(1);
            
            doWithTokens(tokenizer);


  //      }
		
		
	}
	
	private void trackLabel(String label_value) {
		
		String trimmedKey = label_value.trim();
		
		// then we want to track the record label
		if ( this.recordLabels.containsKey( trimmedKey ) ) {
			
			//System.out.println( " size: " + this.recordLabels.size() );
			
			Integer labelID = this.recordLabels.get( trimmedKey ).getFirst();
			Integer countInt = this.recordLabels.get( trimmedKey ).getSecond();
			countInt++;
			
			this.recordLabels.put( trimmedKey, new Pair<>( labelID, countInt ) );
			
		} else {
			
			Integer labelID = this.recordLabels.size();
			this.recordLabels.put( trimmedKey, new Pair<>( labelID, 1 ) );

		//	System.out.println( ">>> Adding Label: '" + trimmedKey + "' @ " + labelID );
			
		}
		
		
	}
	
	public int getNumberOfLabelsSeen() {
		return this.recordLabels.keySet().size();
	}
	
	public Integer getLabelID( String label ) {
		
	//	System.out.println( "getLableID() => '" + label + "' " );
		
		if ( this.recordLabels.containsKey(label) ) {
		
			return this.recordLabels.get( label ).getFirst();
			
		}
		
	//	System.out.println( ".getLabelID() >> returning null with size: " + this.recordLabels.size() );
		return null;
		
	}
	


	/**
	 * Transform the raw record w stats we've learned from the first pass
	 * 
	 * Schema:
	 * 
	 * 		Writable[0]: go dogs, go 1
	 * 		Writable[1]: label_A
	 * 
	 * 1. take the raw record, split off the label and look up its ID from the cache
	 * 2. take the remaining part of the raw record and run it through the 
	 * 
	 */
	@Override
	public void transform(Collection<Writable> vector) {
		
		String label;
		Integer labelID;
		
		String textRecord;
		
		// 1. extract label and sentence from vector
		
		if ( vector.size() != 2 ) {
			// throw some illegal state something here
			return;
		}
		
		textRecord = vector.toArray()[ 0 ].toString();
		label = vector.toArray()[ 1 ].toString();
		
		// 2. get the label ID
		
		labelID = this.getLabelID( label );
		
		// 3. get the converted vector
	//	System.out.print( "Label: " + label + " ");
	//	INDArray tfidfVector = this.convertSentenceToTFIDFVector( textRecord );
		INDArray tfidfVector = this.convertTextRecordToTFIDFVector( textRecord );
		
	//	System.out.println( "cols: " + tfidfVector.columns() );
	//	System.out.println( "rows: " + tfidfVector.rows() );
		
		// 4. rebuild the vector refernece w the schema { vector entries, ..., label }
		
		// 4.a. clear out old entries
		vector.clear();
		
		// 4.b. rebuild
		
		for ( int colID = 0; colID < tfidfVector.columns(); colID++ ) {
			
			vector.add(new DoubleWritable( tfidfVector.getDouble(0, colID) ) );
			
		}
		
		// we always append a label
		
		vector.add(new DoubleWritable( labelID ) );
		
		
	}
	
	


	/**
	 * This is where we'll take the dataset stats learned from the first pass and setup for the 
	 * transform pass
	 * 
	 */
	@Override
	public void evaluateStatistics() {
		// TODO Auto-generated method stub
		
	}
}
