package main.base;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;

/**
 * The class contains helper functions for {@link Booster}.
 */
class Utils {
    private static boolean LIBS_LOADED = false;

    static void loadLibs() {
        if (!LIBS_LOADED) {
            try {
                loadFromJar("lib_lightgbm.so");
                loadFromJar("lib_lightgbm_swig.so");
            } catch (IOException e) {
                throw new UncheckedIOException("Failed to load LightGBM ", e);
            }

            LIBS_LOADED = true;
        }
    }

    static void check(int lgb_return_code, String error_message) {
        if (lgb_return_code == -1) {
            throw new IllegalArgumentException(error_message);
        }

    }

    private static void loadFromJar(String shared_lib_name) throws IOException {
        File temp_file = File.createTempFile("lib", ".so");
        InputStream input_stream = Utils.class.getClassLoader().getResourceAsStream(shared_lib_name);
        OutputStream output_stream = new FileOutputStream(temp_file);

        input_stream.transferTo(output_stream);
        input_stream.close();
        output_stream.close();

        System.load(temp_file.getAbsolutePath());
        temp_file.deleteOnExit();
    }
}