package com.ccc.deeplearning.topicmodeling;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

public class TopicModelingTest {

	@Test
	public void testTopicModeling() throws IOException {
		TopicModeling modeling = new TopicModeling(new File("/home/agibsonccc/Downloads/20news-bydate-train"), 2);
		modeling.train();
		//modeling.dump("/home/agibsonccc/bitcoin/thisiswhatitlookslike");
	}


}
