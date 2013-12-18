package com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf;

import java.util.HashMap;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;

public class Conf extends HashMap<String,String> {


	private static final long serialVersionUID = -7319565384587251582L;

	
	
	public Conf copy() {
		Conf conf = new Conf();
		conf.putAll(this);
		return conf;
	}
	
	@SuppressWarnings("unchecked")
	public Class<? extends BaseMultiLayerNetwork> getClazz(String key) {
		try {
			return (Class<? extends BaseMultiLayerNetwork>) Class.forName(get(key));
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
	}
	
	
	public void put(String key,Object val) {
		put(key,String.valueOf(val));
	}
	
	public Long getLong(String key) {
		String ret = get(key);
		return ret != null ? Long.parseLong(ret) : null;
	}
	
	public Integer getInt(String key) {
		String ret = get(key);
		return ret != null ? Integer.parseInt(ret) : null;
	}
	
	public Double getDouble(String key) {
		String ret = get(key);
		return ret != null ? Double.parseDouble(ret) : null;
	}
	
	public double[] getDoublesWithSeparator(String key,String separator) {
		String get = get(key);
		if(get != null) {
			String[] split = get.split(separator);
			double[] ret = new double[split.length];
			for(int i = 0; i < split.length; i++)
				ret[i] = Double.parseDouble(split[i]);
			return ret;
		}
		return null;
	}

	public int[] getIntsWithSeparator(String key,String separator) {
		String get = get(key);
		if(get != null) {
			get = get.trim();
			String[] split = get.split(separator);
			int[] ret = new int[split.length];
			for(int i = 0; i < split.length; i++)
				ret[i] = Integer.parseInt(split[i]);
			return ret;
		}
		return null;
	}

}
