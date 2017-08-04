package org.deeplearning4j.ui.play.staticroutes;

import com.google.common.net.HttpHeaders;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FilenameUtils;
import org.nd4j.linalg.io.ClassPathResource;
import play.api.libs.MimeTypes;
import play.mvc.Result;

import java.io.InputStream;
import java.util.function.Function;

import static play.mvc.Http.Context.Implicit.response;
import static play.mvc.Results.ok;

/**
 * Simple function for serving assets. Assets are assumed to be in the specified root directory
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Slf4j
public class Assets implements Function<String, Result> {
    private final String assetsRootDirectory;

    @Override
    public Result apply(String s) {
        String fullPath = assetsRootDirectory + s;

        InputStream inputStream;
        try {
            inputStream = new ClassPathResource(fullPath).getInputStream();
        } catch (Exception e) {
            log.debug("Could not find asset: {}", s);
            return ok();
        } catch (Throwable t) {
            return ok();
        }

        String fileName = FilenameUtils.getName(fullPath);

        response().setHeader(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + fileName + "\"");
        scala.Option<String> contentType = MimeTypes.forFileName(fileName);
        String ct;
        if (contentType.isDefined()) {
            ct = contentType.get();
        } else {
            ct = "application/octet-stream";
        }

        return ok(inputStream).as(ct);
    }
}
