package org.deeplearning4j.aws.emr;

import com.amazonaws.services.elasticmapreduce.model.Configuration;

import lombok.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
@Builder
public class EmrConfig {

    protected String classification;
    protected Map<String, String> properties;
    protected List<EmrConfig> configs;

    Configuration toAwsConfig() {
        Configuration config = new Configuration().withClassification(classification).withProperties(properties);
        List<Configuration> subConfigs = new ArrayList<>();
        for (EmrConfig conf : configs){
            subConfigs.add(conf.toAwsConfig());
        }
        return config.withConfigurations(subConfigs);
    }

}
