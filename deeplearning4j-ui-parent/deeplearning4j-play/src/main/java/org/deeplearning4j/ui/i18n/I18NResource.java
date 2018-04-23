package org.deeplearning4j.ui.i18n;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.io.InputStream;

@AllArgsConstructor
@Data
public class I18NResource {

    private final String resource;

    public InputStream getInputStream() throws IOException {
        return new ClassPathResource(resource).getInputStream();
    }


}
