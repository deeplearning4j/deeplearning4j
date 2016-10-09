package org.deeplearning4j.ui.play.staticroutes;

import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.io.ClassPathResource;
import play.mvc.Result;
import play.mvc.Results;

import java.io.File;
import java.io.IOException;
import java.util.function.Function;

import static play.mvc.Results.ok;

/**
 * Simple function for serving assets. Assets are assumed to be in the specified root directory
 *
 * @author Alex Black
 */
@AllArgsConstructor @Slf4j
public class Assets implements Function<String,Result> {
    private final String assetsRootDirectory;

    @Override
    public Result apply(String s) {
        File f;
        try {
            f = new ClassPathResource(assetsRootDirectory + s).getFile();
        } catch (IOException e){
            log.debug("Error for asset request: {}",s,e);
            return Results.notFound("Not found: /assets/" + s);
        }
        return ok(f);
    }
}
