package com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.single.word2vec;

import org.kohsuke.args4j.Option;


public class ActorNetworkRunnerApp extends com.ccc.deeplearning.matrix.jblas.iterativereduce.actor.single.ActorNetworkRunnerApp {


	@Option(name = "-w2vpath",usage="path to the root directory for word2vec to train on")
	protected String word2VecPath;

	public ActorNetworkRunnerApp(String[] args) {
		super(args);
	}



	@Override
	protected void getDataSet() {
		if(dataSet.equals("word")) {

		}
		else
			super.getDataSet();
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
