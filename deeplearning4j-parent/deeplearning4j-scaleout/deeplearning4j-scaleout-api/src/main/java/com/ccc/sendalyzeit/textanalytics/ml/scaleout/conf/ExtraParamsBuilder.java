package com.ccc.sendalyzeit.textanalytics.ml.scaleout.conf;

import java.util.HashMap;
import java.util.Map;

import org.json.JSONObject;

public class ExtraParamsBuilder implements DeepLearningConfigurable {

	private Integer k;
	private Integer epochs;
	private Double learningRate;
	private Double finetuneLearningRate;
	private Double corruptionLevel;
	private Integer finetuneEpochs;
	private String algorithm;
	private Map<String,String> params = new HashMap<String,String>();
	
	
	public ExtraParamsBuilder epochs(Integer epochs) {
		this.epochs = epochs;
		return this;
	}
	
	public ExtraParamsBuilder k(Integer k) {
		this.k = k;
		return this;
	}
	
	public ExtraParamsBuilder learningRate(double learningRate) {
		this.learningRate = learningRate;
		return this;
	}
	
	public ExtraParamsBuilder corruptionlevel(double corruptionLevel) {
		this.corruptionLevel = corruptionLevel;
		return this;
	}
	
	public ExtraParamsBuilder finetuneEpochs(int finetuneEpochs) {
		this.finetuneEpochs = finetuneEpochs;
		return this;
	}
	
	public ExtraParamsBuilder algorithm(String algorithm) {
		this.algorithm = algorithm;
		return this;
	}
	
	
	public ExtraParamsBuilder finetuneLearningRate(double finetuneLearningRate) {
		this.finetuneLearningRate = finetuneLearningRate;
		return this;
	}
	
	public String build() {
		if(k != null)
			params.put(PARAM_K, String.valueOf(k));
		if(learningRate != null)
			params.put(PARAM_LEARNING_RATE,String.valueOf(learningRate));
		if(finetuneLearningRate != null)
			params.put(PARAM_FINETUNE_LR, String.valueOf(finetuneLearningRate));
		if(corruptionLevel != null)
			params.put(PARAM_CORRUPTION_LEVEL, String.valueOf(corruptionLevel));
		if(finetuneEpochs != null)
			params.put(PARAM_FINETUNE_EPOCHS, String.valueOf(finetuneEpochs));
		if(algorithm != null)
			params.put(PARAM_ALGORITHM,algorithm);
		if(epochs != null)
			params.put(PARAM_EPOCHS, String.valueOf(epochs));
		return new JSONObject(params).toString();
	}
	

	
	
	
	
	public void setup(Conf conf) {
		
	}

}
