package com.ccc.deeplearning.word2vec.conf;

import org.json.JSONObject;

import com.ccc.deeplearning.nn.BaseMultiLayerNetwork;


/**
 * Conf capable of handling word2vec 
 * cdbn
 * @author Adaam Gibson
 *
 */
public class Conf extends com.ccc.deeplearning.scaleout.conf.Conf {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2676726056166658343L;
	public final static String WORD2VEC_CDBN = "wordcdbn";
	
	
	
	
	public Conf() {
		super();
		put(PARAM_ALGORITHM,WORD2VEC_CDBN);
		put(CLASS,"com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork");
	}




	@Override
	public Class<? extends BaseMultiLayerNetwork> getClazz(String key) {
		return super.getClazz(key);
	}




	@Override
	public Object[] loadParams(String key) {
		String json = get(key);
		if(json == null)
			return null;
		else {
			JSONObject j = new JSONObject(json);
			String algorithm = j.getString(PARAM_ALGORITHM);
			if(algorithm.equals(PARAM_SDA)) {
				//always present
				Double learningRate = j.getDouble(PARAM_LEARNING_RATE);
				Double corruptionLevel = j.getDouble(PARAM_CORRUPTION_LEVEL);
				Integer epochs = j.getInt(PARAM_EPOCHS);
				if(j.length() > 3) {
					Double finetuneLr = j.getDouble(PARAM_FINETUNE_LR);
					Integer finetuneEpochs = j.getInt(PARAM_FINETUNE_EPOCHS);
					return new Object[]{learningRate,corruptionLevel,epochs,finetuneLr,finetuneEpochs}; 

				}
				else {
					return new Object[] {learningRate,corruptionLevel,epochs};
				}
			}
			else if(algorithm.equals(PARAM_CDBN) || algorithm.equals(PARAM_DBN) || algorithm.equals(WORD2VEC_CDBN)) {
				//always present
				Integer k = j.getInt(PARAM_K);
				if(k < 1)
					throw new IllegalStateException("K must be greater than 0");
				
				Double learningRate = j.getDouble(PARAM_LEARNING_RATE);
				Integer epochs = j.getInt(PARAM_EPOCHS);

				if(j.length() > 3) {
					Double finetuneLr = j.getDouble(PARAM_FINETUNE_LR);
					Integer finetuneEpochs = j.getInt(PARAM_FINETUNE_EPOCHS);
					return new Object[]{k,learningRate,epochs,finetuneLr,finetuneEpochs};

				}
				else {
					return new Object[]{k,learningRate,epochs};
				}
			}
			
			
			
			else if(algorithm.equals(PARAM_RBM) || algorithm.equals(PARAM_CRBM)) {
				//always present
				Integer k = j.getInt(PARAM_K);
				if(k < 1)
					throw new IllegalStateException("K must be greater than 0");
				
				return new Object[]{k};
			}
			else if(algorithm.equals(PARAM_DA)) {
				Double corruptionLevel = j.getDouble(PARAM_CORRUPTION_LEVEL);

				return new Object[]{corruptionLevel};

			}

		}
		
		throw new IllegalStateException("Invalid getter configuration");

	}




	
	public Conf copy() {
		Conf conf = new Conf();
		conf.putAll(this);
		return conf;
	}



}
