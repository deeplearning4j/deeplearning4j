package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.viterbi;

import java.util.*;

import com.ccc.sendalyzeit.textanalytics.util.Window;

public class Datum {

	public final String word;
	public final String label;
	public List<String> features;
	public String guessLabel;
	public String previousLabel;

	public Datum(String word, String label) {
		this.word = word;
		this.label = label;
	}


	public static List<Datum> datums(List<Window> windows,List<String> labels,List<String> previousLabels,List<List<String>> features) {
		List<Datum> ret = new ArrayList<Datum>();
		for(int i = 0; i < windows.size(); i++) {
			ret.add(create(windows.get(i),labels.get(i),previousLabels.get(i),features.get(i)));
		}
		
		
		return ret;
	}

	public static Datum create(Window window,String label,String prevLabel,List<String> features) {
		Datum ret = new Datum(window.getFocusWord(),label);
		ret.previousLabel = prevLabel;
		ret.features = features;
		return ret;
	}

	public static Datum create(Window window,String label,String prevLabel) {
		Datum ret = new Datum(window.getFocusWord(),label);
		ret.previousLabel = prevLabel;
		return ret;
	}


}