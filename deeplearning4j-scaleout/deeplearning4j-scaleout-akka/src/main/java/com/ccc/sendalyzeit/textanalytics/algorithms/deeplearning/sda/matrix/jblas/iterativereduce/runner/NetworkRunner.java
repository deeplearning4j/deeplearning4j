package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.runner;

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

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.ComputableMasterAkka;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.ComputableWorkerAkka;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.Conf;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf.DeepLearningConfigurable;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas.UpdateableMatrix;
import com.google.common.collect.Lists;

public class NetworkRunner implements DeepLearningConfigurable {

	private ComputableMasterAkka master;
	private List<ComputableWorkerAkka> workers = new ArrayList<ComputableWorkerAkka>();
	private Conf conf;
	private int split;
	private ActorSystem system;
	private BaseMultiLayerNetwork output;
	private int epochs;
	private DoubleMatrix input;
	private DoubleMatrix outcomes;
	private static Logger log = LoggerFactory.getLogger(NetworkRunner.class);




	public NetworkRunner(DoubleMatrix input, DoubleMatrix outcomes) {
		this.input = input;
		this.outcomes = outcomes;
	}

	public void setup(Conf conf) {
		this.conf = conf;
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
		c.put(FINE_TUNE_EPOCHS, 100);
		c.put(PRE_TRAIN_EPOCHS,100);
		ret.setup(c);
		return ret;
	}


	public BaseMultiLayerNetwork train()  {
		for(int i = 0; i < epochs; i++) {
			final List<UpdateableMatrix> workerUpdates = new CopyOnWriteArrayList<>();
			final CountDownLatch latch = new CountDownLatch(workers.size());

			for(final ComputableWorkerAkka worker : workers) {
				Future<UpdateableMatrix> future = Futures.future(new Callable<UpdateableMatrix>() {

					@Override
					public UpdateableMatrix call() throws Exception {
						return worker.compute();
					}

				},system.dispatcher());
				
				future.onComplete(new OnComplete<UpdateableMatrix>() {

					@Override
					public void onComplete(Throwable arg0, UpdateableMatrix arg1)
							throws Throwable {
						if(arg0 != null)
							log.error("Error processing worker:",arg0);

						workerUpdates.add(arg1);
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
			UpdateableMatrix masterResult = master.compute(workerUpdates, workerUpdates);

			for(ComputableWorkerAkka worker : workers)
				worker.update(masterResult);


			output = masterResult.get();

		}
		return output;
	}



}
