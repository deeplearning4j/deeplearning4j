package com.ccc.deeplearning.scaleout.core.conf;

import com.ccc.deeplearning.scaleout.conf.DeepLearningConfigurable;

public interface DeepLearningConfigurableDistributed extends DeepLearningConfigurable {

	/* A master url for service discovery*/
	public final static String MASTER_URL = "masterurl";
}
