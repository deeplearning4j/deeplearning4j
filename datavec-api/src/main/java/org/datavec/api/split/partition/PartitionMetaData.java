package org.datavec.api.split.partition;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@NoArgsConstructor
@Builder
@AllArgsConstructor
@Getter
public class PartitionMetaData implements Serializable {
    private int numRecordsUpdated;
    private long sizeUpdated;


}
