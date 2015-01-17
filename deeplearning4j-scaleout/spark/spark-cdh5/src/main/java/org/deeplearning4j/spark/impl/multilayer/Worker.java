package org.deeplearning4j.spark.impl.multilayer;

import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RecordReaderFunction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;


/**
 * This is the prototype DL4J Worker
 *
 * TODO
 * -	figure out how to get hdfs record readers rolling
 * -	separate out worker and master areas of code
 * -	replace ParsePoint with the SVMLight reader
 * -	replace DataPoint with the Pair<INDArray, INDArray>
 *
 *
 * @author josh
 *
 */
public class Worker {

	//private static ParameterVectorUpdateable masterParameterVector = null;

	private static final int D = 10;   // Number of dimensions
	private static final Random rand = new Random(42);

	/**
	 * This is the code run at the "Master" in IterativeReduce parlance
	 */
	static class MasterComputeParameterAverage implements Function2<INDArray, INDArray, INDArray> {
		@Override
		public INDArray call(INDArray worker_update_a, INDArray worker_update_b) {


			String result = new String();

//	    	for (int j = 0; j < D; j++) {
//	        result[j] = a[j] + b[j];
//	       }


			return null;
		}
	}


	static class VectorSum implements Function2<double[], double[], double[]> {
		@Override
		public double[] call(double[] a, double[] b) {
			double[] result = new double[D];
			for (int j = 0; j < D; j++) {
				result[j] = a[j] + b[j];
			}
			return result;
		}
	}


	/**
	 * This is considered the "Worker"
	 * This is the code that will run the .fit() method on the network
	 *
	 * @author josh
	 */
	static class DL4JWorker implements Function<DataSet, INDArray> {

		private final MultiLayerNetwork network;

		DL4JWorker(MultiLayerNetwork network) {
			this.network = network;
		}


		/**
		 * This is the main driver that kicks off the program
		 *
		 * @param args
		 */
		public static void main(String[] args) {

			if (args.length < 2) {
				System.err.println("Usage: DL4J_Spark <file> <iters>");
				System.exit(1);
			}


			// set to test mode
			SparkConf sparkConf = new SparkConf().setAppName("DL4J").setMaster("local");
			JavaSparkContext sc = new JavaSparkContext(sparkConf);
			JavaRDD<String> lines = sc.textFile(args[0]);

			// gotta map this to a Matrix/INDArray
			JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(new SVMLightRecordReader(),Integer.parseInt(args[0]),Integer.parseInt(args[0]))).cache();

			int ITERATIONS = Integer.parseInt(args[1]);
			// Initialize w to a random value

			for (int i = 1; i <= ITERATIONS; i++) {

				System.out.println("On iteration " + i);
/*
	      double[] gradient = points.map(
=======

	      /*double[] gradient = points.map(
>>>>>>> 6ffdca4f9b9039cc7b81a1d12aa400ae84339cb2
	        new DL4JWorker(w)
	      ).reduce( new MasterComputeParameterAverage() );

	      for (int j = 0; j < D; j++) {
	        w[j] -= gradient[j];
<<<<<<< HEAD
	      }
*/
			}
/*
	    val logData = sc.textFile(logFile, 2).cache()
	    	    val numAs = logData.filter(line => line.contains("a")).count()
	    	    val numBs = logData.filter(line => line.contains("b")).count()
	    	    println("Lines with a: %s, Lines with b: %s".format(numAs, numBs))
*/
			//System.out.print("Final w: ");
			//printWeights(w);
			sc.stop();
		}

		@Override
		public INDArray call(DataSet v1) throws Exception {
			network.fit(v1);
			return network.params();
		}
	}
}