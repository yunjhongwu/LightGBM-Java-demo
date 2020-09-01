package main.base;

import java.util.Map;
import java.util.Map.Entry;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_int;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_long_long;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;

/**
 * The class is a LightGBM wrapper.
 */
public class Booster {
    static {
        Utils.loadLibs();
    }
    private final int num_iterations;
    private String params_str;
    private SWIGTYPE_p_void swig_booster = null;
    private SWIGTYPE_p_p_void swig_out_booster = null;

    public Booster(int num_iterations, Map<String, Object> params) {
        if (params == null) {
            throw new IllegalArgumentException("Parameter specification is null.");
        }
        this.num_iterations = num_iterations;
        this.params_str = getParameterString(num_iterations, params);
    }

    /**
     * Load LightGBM Booster from a text file.
     * 
     * @param filepath file name and path of a LightGBM model
     */
    public Booster(String model_str) {
        swig_out_booster = lightgbmlib.voidpp_handle();
        SWIGTYPE_p_int num_iter_out = lightgbmlib.new_intp();
        Utils.check(lightgbmlib.LGBM_BoosterLoadModelFromString(model_str, num_iter_out, swig_out_booster),
                "Booster creation failed");
        num_iterations = lightgbmlib.int32_tp_value(num_iter_out);
        swig_booster = lightgbmlib.voidpp_value(swig_out_booster);
    }

    /**
     * Train a LightGBM model with respect to input dataset.
     * 
     * @param data training data
     */
    public void train(Dataset data) {
        reset();

        swig_out_booster = lightgbmlib.voidpp_handle();
        SWIGTYPE_p_void data_handle = data.getDataHandle(params_str, true);
        Utils.check(lightgbmlib.LGBM_BoosterCreate(data_handle, params_str, swig_out_booster),
                "Booster creation failed while initializing training process.");
        swig_booster = lightgbmlib.voidpp_value(swig_out_booster);
        trainBooster(swig_booster);
    }

    /**
     * Generate prediction from a trained LightGBM model based on input dataset.
     * 
     * @param data test data
     */
    public float[] predict(Dataset data) {
        if (swig_booster == null) {
            throw new IllegalStateException("Booster has not been trained.");
        } else {
            SWIGTYPE_p_double swig_out_score = lightgbmlib.new_doubleArray(data.getNumOfInstances());
            SWIGTYPE_p_long_long len_long_ptr = lightgbmlib.new_int64_tp();
            lightgbmlib.int64_tp_assign(len_long_ptr, data.getNumOfInstances() * data.getNumOfFeatures());

            SWIGTYPE_p_void data_handle = data.getDataHandle(params_str, false);
            Utils.check(lightgbmlib.LGBM_BoosterPredictForMat(swig_booster, data_handle,
                    lightgbmlibConstants.C_API_DTYPE_FLOAT32, data.getNumOfInstances(), data.getNumOfFeatures(), 1,
                    lightgbmlibConstants.C_API_PREDICT_NORMAL, -1, num_iterations, "verbosity=-1", len_long_ptr,
                    swig_out_score), "Booster prediction failed.");

            float[] outputs = new float[data.getNumOfInstances()];

            for (int i = 0; i < outputs.length; i++) {
                outputs[i] = (float) lightgbmlib.doubleArray_getitem(swig_out_score, i);
            }
            lightgbmlib.delete_int64_tp(len_long_ptr);
            lightgbmlib.delete_doubleArray(swig_out_score);

            return outputs;
        }

    }

    /**
     * Return a LightGBM model as string.
     * 
     * @return model string
     */
    public String getModelString() {
        if (swig_booster == null) {
            throw new IllegalStateException("Booster has not been trained.");
        } else {
            return lightgbmlib.LGBM_BoosterSaveModelToStringSWIG(swig_booster, 0, -1, 1,
                    lightgbmlib.C_API_FEATURE_IMPORTANCE_SPLIT, lightgbmlib.new_int64_tp());
        }
    }

    private void reset() {
        lightgbmlib.delete_voidpp(swig_out_booster);
        lightgbmlib.LGBM_BoosterFree(swig_booster);
        swig_booster = null;
        swig_out_booster = null;
    }

    private String getParameterString(int num_iterations, Map<String, Object> params) {
        StringBuilder params_builder = new StringBuilder("num_iterations=" + num_iterations + " ");
        for (Entry<String, Object> param : params.entrySet()) {
            if (!param.getKey().equals("num_iterations")) {
                params_builder.append(param.getKey());
                params_builder.append('=');
                params_builder.append(param.getValue());
                params_builder.append(' ');
            }
        }

        params_builder.deleteCharAt(params_builder.length() - 1);

        return params_builder.toString();
    }

    private void trainBooster(SWIGTYPE_p_void swig_booster) {
        final SWIGTYPE_p_int swig_finished_indicator = lightgbmlib.new_intp();

        try {
            for (int niter = 0; (niter < num_iterations)
                    && (lightgbmlib.intp_value(swig_finished_indicator) != 1); niter++) {
                Utils.check(lightgbmlib.LGBM_BoosterUpdateOneIter(swig_booster, swig_finished_indicator),
                        "Booster training failed at iteration " + niter + ".");

            }
        } finally {
            lightgbmlib.delete_intp(swig_finished_indicator);
        }
    }

}