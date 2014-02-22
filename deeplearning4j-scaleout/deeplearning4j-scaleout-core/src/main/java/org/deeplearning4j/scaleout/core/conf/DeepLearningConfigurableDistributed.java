package org.deeplearning4j.scaleout.core.conf;

import org.deeplearning4j.scaleout.conf.DeepLearningConfigurable;

public interface DeepLearningConfigurableDistributed extends DeepLearningConfigurable {

	/* A master url for service discovery*/
	public final static String MASTER_URL = "masterurl";
}
