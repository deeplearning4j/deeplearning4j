package com.ccc.deeplearning.word2vec.viterbi;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;

import com.ccc.deeplearning.berkeley.CounterMap;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.util.MatrixUtil;
import com.ccc.deeplearning.word2vec.util.Window;
/**
 * 
 * @author Adam Gibson
 *
 */
public class ViterbiUtil {


	public static DoubleMatrix fromWindows(List<Window> window,List<String> index) {
		Index labelIndex = fromLabels(index);
		return fromWindows(window,labelIndex);
	}
	
	
	public static DoubleMatrix fromWindows(List<Window> window,Index labelIndex) {
		DoubleMatrix ret = new DoubleMatrix(window.size(),labelIndex.size());
		for(int i = 0; i < ret.rows; i++) {
			ret.putRow(i, MatrixUtil.toOutcomeVector(labelIndex.indexOf(window.get(i).getLabel()), labelIndex.size()));
		}
		return ret;
	}
	
	
	public static Index fromLabels(List<String> labels) {
		Index labelIndex = new Index();
		labelIndex.add("NONE");
		for(int i = 0 ; i < labels.size(); i++)
			labelIndex.add(labels.get(i));
		return labelIndex;
	}
	
	public static void incrementTransitionProbabilities(CounterMap<Integer,Integer> counter,DoubleMatrix d) {
		for(int i = 1; i < d.rows; i++) {
			counter.incrementCount(SimpleBlas.iamax(d.getRow(i - 1)),SimpleBlas.iamax(d.getRow(i)), 1.0);
		}

	}
	

	public static Viterbi initializeOnData(Index labels,DataSet d) {
		CounterMap<Integer,Integer> transitionProbabilities = new CounterMap<>();


		incrementTransitionProbabilities(transitionProbabilities,d.getSecond());




		DoubleMatrix transitionProbabilities2 = CounterUtil.convert(transitionProbabilities);

		Index labelIndex = new Index();
		labelIndex.add("NONE");
		for(int i = 0 ; i < labels.size(); i++)
			labelIndex.add(labels.get(i));

		//the output single labelStrings are input to the sequence classifier for viterbi
		return new Viterbi(labelIndex,ViterbiUtil.featureIndexFromLabelIndex(labelIndex),transitionProbabilities2);
	}
	
	/**
	 * Converts a double matrix of outcomes to a 2d list of strings
	 * @param matrix the matrix of outcomes to convert
	 * @return a 2d list of strings with the equivalent numbers
	 * of the matrix
	 */
	public static List<List<String>> toFeatures(DoubleMatrix matrix) {
		List<List<String>> ret = new ArrayList<List<String>>();
		for(int i = 0; i < matrix.rows; i++) {
			List<String> row = new ArrayList<String>();
			for(int j = 0; j < matrix.columns; j++)
				row.add(String.valueOf(matrix.get(i,j)));
			ret.add(row);
		}
		return ret;
	}

	/**
	 * Creates an index from the label index.
	 * This is meant to represent an index list of
	 * labelStrings.
	 *  
	 * @param labelIndex the index to convert
	 * @return an index where each label is a number
	 */
	public static Index featureIndexFromLabelIndex(Index labelIndex) {
		Index ret = new Index();
		for(int i = 0; i < labelIndex.size(); i++)
			ret.add(String.valueOf(i));
		return ret;
	}


	public static List<Datum> previousLabelDatums(List<Datum> data) {
		// this is so that the feature factory code doesn't accidentally use the
		// true label info
		List<Datum> newData = new ArrayList<Datum>();
		List<String> words = new ArrayList<String>();
		List<String> labels = new ArrayList<String>();
		Map<String, Integer> labelIndex = new HashMap<String, Integer>();

		for (Datum datum : data) {
			words.add(datum.word);
			if (labelIndex.containsKey(datum.label) == false) {
				labelIndex.put(datum.label, labels.size());
				labels.add(datum.label);
			}
		}

		// compute features for all possible previous labelStrings in advance for
		// Viterbi algorithm
		for (int i = 0; i < data.size(); i++) {
			Datum datum = data.get(i);

			if (i == 0) {
				String previousLabel = "O";
				datum.features = datum.features;

				Datum newDatum = new Datum(datum.word, datum.label);
				newDatum.features = datum.features;
				newDatum.previousLabel = previousLabel;
				newData.add(newDatum);

			} else {
				for (String previousLabel : labels) {
					datum.features = data.get(i - 1).features;
					Datum newDatum = new Datum(datum.word, datum.label);
					newDatum.features = data.get(i - 1).features;
					newDatum.previousLabel = previousLabel;
					newData.add(newDatum);
				}
			}

		}

		return newData;
	}



}