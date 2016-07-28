package org.deeplearning4j.nn.conf.layers;

import lombok.*;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public class BaseRecurrentLayer extends FeedForwardLayer {
	
	protected BaseRecurrentLayer(Builder builder){
		super(builder);
	}
	
	@AllArgsConstructor
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<Builder<T>> {

    }
}
