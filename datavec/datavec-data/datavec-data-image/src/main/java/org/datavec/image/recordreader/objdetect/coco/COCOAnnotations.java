package org.datavec.image.recordreader.objdetect.coco;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.ArrayList;


@Data
@Builder
@NoArgsConstructor
public class COCOAnnotations extends ArrayList<COCOAnnotation> {
}
