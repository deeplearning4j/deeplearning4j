package com.ccc.sendalyzeit.textanalytics.algorithms.datasets;

import java.io.*;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.MnistDataFetcher;
import com.ccc.sendalyzeit.textanalytics.util.MathUtils;

/**
 * A data set (example/outcome pairs)
 * The outcomes are specifically for neural network encoding such that
 * any labels that are considered true are 1s. The rest are zeros.
 * @author Adam Gibson
 *
 */
public class DataSet extends Pair<DoubleMatrix,DoubleMatrix> {

	private static final long serialVersionUID = 1935520764586513365L;
	
	public DataSet(Pair<DoubleMatrix,DoubleMatrix> pair) {
		this(pair.getFirst(),pair.getSecond());
	}
	
	public DataSet(DoubleMatrix first, DoubleMatrix second) {
		super(first, second);
	}

	
	public DoubleMatrix exampleSums() {
		return getFirst().columnSums();
	}
	
	public DoubleMatrix exampleMaxs() {
		return getFirst().columnMaxs();
	}
	
	public DoubleMatrix exampleMeans() {
		return getFirst().columnMeans();
	}
	
	public void saveTo(File file,boolean binary) throws IOException {
		if(file.exists())
			file.delete();
		file.createNewFile();
		
		if(binary) {
			DataOutputStream bos = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(file)));
			getFirst().out(bos);
			getSecond().out(bos);
			bos.flush();
			bos.close();
			
		}
		else {
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
			for(int i = 0; i < numExamples(); i++) {
				bos.write(getFirst().getRow(i).toString("%.3f", "[", "]", ", ", ";").getBytes());
				bos.write("\t".getBytes());
				bos.write(getSecond().getRow(i).toString("%.3f", "[", "]", ", ", ";").getBytes());
				bos.write("\n".getBytes())	;

				
			}
			bos.flush();
			bos.close();
			
		}
	}
	
	
	public static DataSet load(File path) throws IOException {
		DataInputStream bis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)));
		DoubleMatrix x = new DoubleMatrix(1,1);
		DoubleMatrix y = new DoubleMatrix(1,1);
		x.in(bis);
		y.in(bis);
		bis.close();
		return new DataSet(x,y);
	}
	
	/**
	 * Sample without replacement and a random rng
	 * @param numSamples the number of samples to get
	 * @return a sample data set without replacement
	 */
	public DataSet sample(int numSamples) {
		return sample(numSamples,new MersenneTwister(System.currentTimeMillis()));
	}
	
	/**
	 * Sample without replacement
	 * @param numSamples the number of samples to get
	 * @param rng the rng to use
	 * @return the sampled dataset without replacement
	 */
	public DataSet sample(int numSamples,RandomGenerator rng) {
		return sample(numSamples,rng,false);
	}
	
	/**
	 * Sample a dataset numSamples times
	 * @param numSamples the number of samples to get
	 * @param withReplacement the rng to use
	 * @return the sampled dataset without replacement
	 */
	public DataSet sample(int numSamples,boolean withReplacement) {
		return sample(numSamples,new MersenneTwister(System.currentTimeMillis()),withReplacement);
	}
	
	/**
	 * Sample a dataset
	 * @param numSamples the number of samples to get
	 * @param rng the rng to use
	 * @param withReplacement whether to allow duplicates (only tracked by example row number)
	 * @return the sample dataset
	 */
	public DataSet sample(int numSamples,RandomGenerator rng,boolean withReplacement) {
		if(numSamples >= numExamples())
			return this;
		else {
			DoubleMatrix examples = new DoubleMatrix(numSamples,getFirst().columns);
			DoubleMatrix outcomes = new DoubleMatrix(numSamples,numOutcomes());
			Set<Integer> added = new HashSet<Integer>();
			for(int i = 0; i < numSamples; i++) {
				int picked = rng.nextInt(numExamples());
				while(added.contains(picked)) {
					picked = rng.nextInt(numExamples());

				}
				examples.putRow(i,getFirst().getRow(i));
				outcomes.putRow(i,getSecond().getRow(i));

			}
			return new DataSet(examples,outcomes);
		}
	}

	public void roundToTheNearest(int roundTo) {
		for(int i = 0; i < getFirst().length; i++) {
			double curr = getFirst().get(i);
			getFirst().put(i,MathUtils.roundDouble(curr, roundTo));
		}
	}
	
	public int numOutcomes() {
		return getSecond().columns;
	}

	public int numExamples() {
		return getFirst().rows;
	}

	
	public static void main(String[] args) throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(100);
		DataSet write = new DataSet(fetcher.next());
		write.saveTo(new File(args[0]), false);
		
		
	}

}
