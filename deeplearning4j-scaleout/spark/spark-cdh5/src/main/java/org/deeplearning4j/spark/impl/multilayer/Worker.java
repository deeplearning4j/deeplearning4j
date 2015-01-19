package org.deeplearning4j.spark.impl.multilayer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.canova.api.split.StringSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RecordReaderFunction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;


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




	/**
	 * This is the main driver that kicks off the program
	 *
	 * Current best idea: 
	 *
	 * 		http://stackoverflow.com/questions/23402303/apache-spark-moving-average
	 *
	 * @param args
	 */
	public static void main(String[] args) {

		System.err.println("running worker..");
		if (args.length < 2) {
			System.err.println("Usage: DL4J_Spark <file> <iters>");
			System.exit(1);
		}

		System.err.println("Setting up Spark Conf");

		// set to test mode
		SparkConf sparkConf = new SparkConf()
				.setMaster("spark://localhost")
				.setAppName("SparkDebugExample");

		System.out.println("Setting up Spark Context...");

		JavaSparkContext sc = new JavaSparkContext(sparkConf);
		JavaRDD<String> lines = sc.textFile(args[0]);
        long count = lines.count();

		// gotta map this to a Matrix/INDArray
		JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(new SVMLightRecordReader(), Integer.parseInt(args[2]), Integer.parseInt(args[3])));

		points.map(new Function<DataSet, Object>() {
			@Override
			public Object call(DataSet v1) throws Exception {
				System.out.println(v1.getFeatureMatrix());
				return null;
			}
		}).cache();
		int ITERATIONS = Integer.parseInt(args[1]);
		// Initialize w to a random value

		long c = lines.count();
		System.out.println("svmLight records: " + c);

		for (int i = 1; i <= ITERATIONS; i++) {

			System.out.println("On iteration " + i);
			//INDArray out = points.map( new DL4JWorker() );

			System.out.println("end iteration " + i);


			sc.stop();
		}


	}
}