package org.deeplearning4j.datasets;

import java.io.*;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.Persistable;
import org.deeplearning4j.util.MathUtils;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

/**
 * A data set (example/outcome pairs)
 * The outcomes are specifically for neural network encoding such that
 * any labels that are considered true are 1s. The rest are zeros.
 * @author Adam Gibson
 *
 */
public class DataSet extends Pair<DoubleMatrix,DoubleMatrix> implements Persistable,Iterable<DataSet> {

	private static final long serialVersionUID = 1935520764586513365L;
	private static Logger log = LoggerFactory.getLogger(DataSet.class);


	public DataSet() {
		this(DoubleMatrix.zeros(1),DoubleMatrix.zeros(1));
	}

	public DataSet(Pair<DoubleMatrix,DoubleMatrix> pair) {
		this(pair.getFirst(),pair.getSecond());
	}

	public DataSet(DoubleMatrix first, DoubleMatrix second) {
		super(first, second);
		if(first.rows != second.rows)
			throw new IllegalStateException("Invalid data set; first and second do not have equal rows. First was " + first.rows + " second was " + second.rows);


	}

	public DataSetIterator iterator(int batches) {
		List<DataSet> list = this.dataSetBatches(batches);
		return new ListDataSetIterator(list);
	}


	public DataSet copy() {
		return new DataSet(getFirst(),getSecond());
	}




	public static DataSet empty() {
		return new DataSet(DoubleMatrix.zeros(1),DoubleMatrix.zeros(1));
	}

	public static DataSet merge(List<DataSet> data) {
		if(data.isEmpty())
			throw new IllegalArgumentException("Unable to merge empty dataset");
		DataSet first = data.get(0);
		int numExamples = totalExamples(data);
		DoubleMatrix in = new DoubleMatrix(numExamples,first.getFirst().columns);
		DoubleMatrix out = new DoubleMatrix(numExamples,first.getSecond().columns);
		int count = 0;

		for(int i = 0; i < data.size(); i++) {
			DataSet d1 = data.get(i);
			for(int j = 0; j < d1.numExamples(); j++) {
				DataSet example = d1.get(j);
				in.putRow(count,example.getFirst());
				out.putRow(count,example.getSecond());
				count++;
			}


		}
		return new DataSet(in,out);
	}


	public void shuffle() {
		List<DataSet> list = asList();
		Collections.shuffle(list);
		DataSet ret = DataSet.merge(list);
		setFirst(ret.getFirst());
		setSecond(ret.getSecond());
	}

	
	public void normalize() {
		MatrixUtil.normalizeMatrix(getFirst());
	}
	

	private static int totalExamples(Collection<DataSet> coll) {
		int count = 0;
		for(DataSet d : coll)
			count += d.numExamples();
		return count;
	}



	public int numInputs() {
		return getFirst().columns;
	}

	public void validate() {
		if(getFirst().rows != getSecond().rows)
			throw new IllegalStateException("Invalid dataset");
	}

	public int outcome() {
		if(this.numExamples() > 1)
			throw new IllegalStateException("Unable to derive outcome for dataset greater than one row");
		return SimpleBlas.iamax(getSecond());
	}

	
	/**
	 * Clears the outcome matrix setting a new number of labels
	 * @param labels the number of labels/columns in the outcome matrix
	 * Note that this clears the labels for each example
	 */
	public void setNewNumberOfLabels(int labels) {
		int examples = numExamples();
		DoubleMatrix newOutcomes = new DoubleMatrix(examples,labels);
		setSecond(newOutcomes);
	}
	
	/**
	 * Sets the outcome of a particular example
	 * @param example the example to set
	 * @param label the label of the outcome
	 */
	public void setOutcome(int example,int label) {
		if(example > numExamples())
			throw new IllegalArgumentException("No example at " + example);
		if(label > numOutcomes() || label < 0)
			throw new IllegalArgumentException("Illegal label");
		
		DoubleMatrix outcome = MatrixUtil.toOutcomeVector(label, numOutcomes());
		getSecond().putRow(example,outcome);
	}
	
	/**
	 * Gets a copy of example i
	 * @param i the example to get
	 * @return the example at i (one example)
	 */
	public DataSet get(int i) {
		if(i > numExamples() || i < 0)
			throw new IllegalArgumentException("invalid example number");
		
		return new DataSet(getFirst().getRow(i),getSecond().getRow(i));
	}

	public List<List<DataSet>> batchBy(int num) {
		return Lists.partition(asList(),num);
	}


	/**
	 * Gets the label distribution (counts of each possible outcome)
	 * @return the counts of each possible outcome
	 */
	public Counter<Integer> outcomeCounts() {
		List<DataSet> list = asList();
		Counter<Integer> ret = new Counter<>();
		for(int i = 0; i < list.size(); i++) {
			ret.incrementCount(list.get(i).outcome(),1.0);
		}
		return ret;
	}


	public DataSet filterBy(int[] labels) {
		List<DataSet> list = asList();
		List<DataSet> newList = new ArrayList<DataSet>();
		List<Integer> labelList = new ArrayList<Integer>();
		for(int i : labels)
			labelList.add(i);
		for(DataSet d : list) {
			if(labelList.contains(d.getLabel(d))) {
				newList.add(d);
			}
		}

		return DataSet.merge(newList);
	}




	/**
	 * Partitions the data set by the specified number.
	 * @param num the number to split by
	 * @return the paritioned data set
	 */
	public List<DataSet> dataSetBatches(int num) {
		List<List<DataSet>> list =  Lists.partition(asList(),num);
		List<DataSet> ret = new ArrayList<>();
		for(List<DataSet> l : list)
			ret.add(DataSet.merge(l));
		return ret;

	}


	/**
	 * Sorts the dataset by label:
	 * Splits the data set such that examples are sorted by their labels.
	 * A ten label dataset would produce lists with batches like the following:
	 * x1   y = 1
	 * x2   y = 2
	 * ...
	 * x10  y = 10
	 * @return a list of data sets partitioned by outcomes
	 */
	public List<List<DataSet>> sortAndBatchByNumLabels() {
		sortByLabel();
		return Lists.partition(asList(),numOutcomes());
	}

	public List<List<DataSet>> batchByNumLabels() {
		return Lists.partition(asList(),numOutcomes());
	}


	public List<DataSet> asList() {
		List<DataSet> list = new ArrayList<DataSet>(numExamples());
		for(int i = 0; i < numExamples(); i++)  {
			list.add(new DataSet(getFirst().getRow(i),getSecond().getRow(i)));
		}
		return list;
	}

	public Pair<DataSet,DataSet> splitTestAndTrain(int numHoldout) {

		if(numHoldout >= numExamples())
			throw new IllegalArgumentException("Unable to split on size larger than the number of rows");


		List<DataSet> list = asList();

		Collections.rotate(list, 3);
		Collections.shuffle(list);
		List<List<DataSet>> partition = new ArrayList<List<DataSet>>();
		partition.add(list.subList(0, numHoldout));
		partition.add(list.subList(numHoldout, list.size()));
		DataSet train = merge(partition.get(0));
		DataSet test = merge(partition.get(1));
		return new Pair<>(train,test);
	}

	/**
	 * Organizes the dataset to minimize sampling error
	 * while still allowing efficient batching.
	 */
	public void sortByLabel() {
		Map<Integer,Queue<DataSet>> map = new HashMap<Integer,Queue<DataSet>>();
		List<DataSet> data = asList();
		int numLabels = numOutcomes();
		int examples = numExamples();
		for(DataSet d : data) {
			int label = getLabel(d);
			Queue<DataSet> q = map.get(label);
			if(q == null) {
				q = new ArrayDeque<DataSet>();
				map.put(label, q);
			}
			q.add(d);
		}

		for(Integer label : map.keySet()) {
			log.info("Label " + label + " has " + map.get(label).size() + " elements");
		}

		//ideal input splits: 1 of each label in each batch
		//after we run out of ideal batches: fall back to a new strategy
		boolean optimal = true;
		for(int i = 0; i < examples; i++) {
			if(optimal) {
				for(int j = 0; j < numLabels; j++) {
					Queue<DataSet> q = map.get(j);
					DataSet next = q.poll();
					//add a row; go to next
					if(next != null) {
						addRow(next,i);
						i++;
					}
					else {
						optimal = false;
						break;
					}
				}
			}
			else {
				DataSet add = null;
				for(Queue<DataSet> q : map.values()) {
					if(!q.isEmpty()) {
						add = q.poll();
						break;
					}
				}

				addRow(add,i);

			}


		}


	}


	public void addRow(DataSet d, int i) {
		if(i > numExamples() || d == null)
			throw new IllegalArgumentException("Invalid index for adding a row");
		getFirst().putRow(i, d.getFirst());
		getSecond().putRow(i,d.getSecond());
	}


	private int getLabel(DataSet data) {
		return SimpleBlas.iamax(data.getSecond());
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
				if(!withReplacement)
					while(added.contains(picked)) {
						picked = rng.nextInt(numExamples());

					}
				examples.putRow(i,get(picked).getFirst());
				outcomes.putRow(i,get(picked).getSecond());

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




	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("===========INPUT===================\n")
		.append(getFirst().toString().replaceAll(";","\n"))
		.append("\n=================OUTPUT==================\n")
		.append(getSecond().toString().replaceAll(";","\n"));
		return builder.toString();
	}

	public static void main(String[] args) throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(100);
		DataSet write = new DataSet(fetcher.next());
		write.saveTo(new File(args[0]), false);


	}

	@Override
	public void write(OutputStream os) {
		DataOutputStream dos = new DataOutputStream(os);

		try {
			getFirst().out(dos);
			getSecond().out(dos);

		} catch (IOException e) {
			throw new RuntimeException(e);
		}

	}

	@Override
	public void load(InputStream is) {
		DataInputStream dis = new DataInputStream(is);
		try {
			getFirst().in(dis);
			getSecond().in(dis);

		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public Iterator<DataSet> iterator() {
		return asList().iterator();
	}

}
