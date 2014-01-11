package com.ccc.deeplearning.word2vec.iterativereduce.actor.multilayer;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.kohsuke.args4j.Option;

import com.ccc.deeplearning.word2vec.Word2Vec;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIterator;
import com.ccc.deeplearning.word2vec.iterator.Word2VecDataSetIteratorImpl;
import com.ccc.deeplearning.word2vec.loader.Word2VecLoader;
import com.ccc.deeplearning.word2vec.util.Window;


public class ActorNetworkRunnerApp extends com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.multilayer.ActorNetworkRunnerApp {

	@Option(name = "-w2vpath",usage="path to the root directory for word2vec to train on")
	protected String word2VecPath;
	@Option(name="-labels",usage="comma separated list of labels to use for labeling sequences of text")
	protected String labels;
	@Option(name="-path",usage="path to training dataset")
	protected String trainingPath;
	
	private ActorNetworkRunner runner;
	
	private Word2VecDataSetIterator iter;
	
	
	public ActorNetworkRunnerApp(String[] args) {
		super(args);
	}

	@Override
	protected void getDataSet() {
		if(dataSet.equals("word")) {
			Word2Vec vec;
			try {
				vec = Word2VecLoader.loadModel(new File(word2VecPath));
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
			List<String> labels2 = Arrays.asList(labels.split(","));
			if(labels2.isEmpty())
				throw new IllegalArgumentException("Please specify a label");
			if(!labels2.get(0).equals("NONE")) {
				List<String> withNone = new ArrayList<String>();
				withNone.add("NONE");
				withNone.addAll(labels2);
				labels2 = withNone;
			}
			
			
			this.iter = new Word2VecDataSetIteratorImpl(trainingPath,labels2,split, vec);

		}
		else
			super.getDataSet();
	}

	
	
	
	

	@Override
	public void train() {
		List<Window> data;
		if(!iter.hasNext()) {
			throw new IllegalStateException("Unable to train on empty dataset");
		}
		data = iter.next();
		
		runner.train(data);
		
	}

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		ActorNetworkRunnerApp app = new ActorNetworkRunnerApp(args);
		app.exec();
		if(app.type.equals("master"))
			app.train();
	}

}
