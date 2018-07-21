package org.nd4j.api.loader;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.nd4j.api.loader.Source;

import java.io.*;

/**
 * A simple {@link Source} that returns a (buffered) FileInputStream for the specified file path
 */
@AllArgsConstructor
public class LocalFileSource implements Source {
    @Getter
    private String path;

    @Override
    public InputStream getInputStream() throws IOException {
        return new BufferedInputStream(new FileInputStream(new File(path)));
    }
}
