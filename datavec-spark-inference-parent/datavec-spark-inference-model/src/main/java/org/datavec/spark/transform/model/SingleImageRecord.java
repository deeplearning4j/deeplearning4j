package org.datavec.spark.transform.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.net.URI;

/**
 * Created by kepricon on 17. 5. 24.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class SingleImageRecord {
    private URI uri;
}
