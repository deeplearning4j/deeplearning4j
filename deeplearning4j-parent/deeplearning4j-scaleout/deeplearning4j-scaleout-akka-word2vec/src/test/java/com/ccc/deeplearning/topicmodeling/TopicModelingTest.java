package com.ccc.deeplearning.topicmodeling;

import java.io.IOException;

import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

public class TopicModelingTest {

	@Test
	public void testTopicModeling() throws IOException {
		TopicModeling modeling = new TopicModeling(new ClassPathResource("/bitcoin/topics").getFile(), 2, true);
		modeling.train();
		modeling.dump("/home/agibsonccc/bitcoin/thisiswhatitlookslike");
	}


}
