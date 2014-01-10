package com.ccc.deeplearning.iterativereduce.akka.runner;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import scala.concurrent.Future;

import akka.actor.ActorSystem;
import akka.dispatch.Futures;
import akka.dispatch.OnComplete;

import com.ccc.deeplearning.iterativereduce.akka.ComputableMasterAkka;
import com.ccc.deeplearning.iterativereduce.akka.ComputableWorkerAkka;
import com.ccc.deeplearning.iterativereduce.akka.DeepLearningAccumulator;
import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;
import com.ccc.deeplearning.scaleout.conf.Conf;
import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;
import com.ccc.deeplearning.scaleout.iterativereduce.multi.UpdateableImpl;
import com.google.common.collect.Lists;

public class NetworkRunner implements DeepLearningConfigurable {

	private ComputableMasterAkka master;
	private List<ComputableWorkerAkka> workers = new ArrayList<ComputableWorkerAkka>();
	private Conf conf;
	private int split;
	private ActorSystem system;
	private DeepLearningAccumulator acc;
	private BaseMultiLayerNetwork output;
	private int epochs;
	private DoubleMatrix input;
	private DoubleMatrix outcomes;
	private static Logger log = LoggerFactory.getLogger(NetworkRunner.class);
	private boolean setup;



	public NetworkRunner() {
		acc = new DeepLearningAccumulator();
	}

	public void setup(Conf conf) {
		this.conf = conf;
	}

	private void doSetup(Conf conf) {
		split = conf.getInt(SPLIT);
		system = ActorSystem.create();
		conf.put(N_IN, String.valueOf(input.columns));
		conf.put(OUT, String.valueOf(outcomes.columns));
		epochs = conf.getInt(FINE_TUNE_EPOCHS);
		master = new ComputableMasterAkka();
		master.setup(this.conf);
		if(conf.get(SEED) != null)
			org.jblas.util.Random.seed(Integer.parseInt(conf.get(SEED)));


		List<Integer> rows2 = new ArrayList<>();

		for(int i = 0; i < input.rows; i++)
			rows2.add(i);
		List<List<Integer>> indices = Lists.partition(rows2, split);
		for(int i = 0; i < indices.size(); i++) 
			workers.add(fromMatrices(convert(indices.get(i))));
		setup = true;

	}

	private int[] convert(List<Integer> ints) {
		int[] ret = new int[ints.size()];
		for(int i = 0; i < ints.size(); i++)
			ret[i] = ints.get(i);
		return ret;
	}

	private ComputableWorkerAkka fromMatrices(int[] rows) {
		ComputableWorkerAkka ret =  new ComputableWorkerAkka(input,outcomes, rows);
		Conf c = conf.copy();
		c.put(ROWS, rows.length);
		ret.setup(c);
		return ret;
	}


	public BaseMultiLayerNetwork train(DoubleMatrix input,DoubleMatrix labels)  {
		this.input = input;
		this.outcomes = labels;
		if(!setup)
			doSetup(conf);

		for(int i = 0; i < epochs; i++) {
			final List<UpdateableImpl> workerUpdates = new CopyOnWriteArrayList<>();
			final CountDownLatch latch = new CountDownLatch(workers.size());
			final int epoch = i + 1;
			for(final ComputableWorkerAkka worker : workers) {
				
				Future<UpdateableImpl> future = Futures.future(new Callable<UpdateableImpl>() {

					@Override
					public UpdateableImpl call() throws Exception {
						return worker.compute();
					}

				},system.dispatcher());

				future.onComplete(new OnComplete<UpdateableImpl>() {

					@Override
					public void onComplete(Throwable arg0, UpdateableImpl arg1)
							throws Throwable {
						if(arg0 != null)
							log.error("Error processing worker:",arg0);
						workerUpdates.add(arg1);
						log.info("Worker updates error is " + arg1.get().negativeLogLikelihood() + " for epoch " + epoch + " and number of updates so far " + workerUpdates.size());

						latch.countDown();
						worker.incrementIteration();
					}

				}, system.dispatcher());
			} 

			try {
				latch.await();
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
			}


			log.info("Computation for iteration " + i + " done"	);
			UpdateableImpl masterResult = master.compute(workerUpdates, workerUpdates);

			for(ComputableWorkerAkka worker : workers)
				worker.update(masterResult);


			output = masterResult.get();
			acc.accumulate(output);
		}
		return output;
	}
	
	public BaseMultiLayerNetwork result()  {
		return acc.averaged();
	}
	

	public DoubleMatrix getInput() {
		return input;
	}

	public void setInput(DoubleMatrix input) {
		this.input = input;
	}

	public DoubleMatrix getOutcomes() {
		return outcomes;
	}

	public void setOutcomes(DoubleMatrix outcomes) {
		this.outcomes = outcomes;
	}



}
