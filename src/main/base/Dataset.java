package main.base;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_float;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_p_void;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;

/**
 * The class maintains a dataset for training/predicting with {@link Booster}.
 */
public class Dataset {
    static {
        Utils.loadLibs();
        lightgbmlib.LGBM_DatasetCreateFromMat(lightgbmlib.float_to_voidp_ptr(lightgbmlib.new_floatArray(1)),
                lightgbmlibConstants.C_API_DTYPE_FLOAT32, 1, 1, 1, "verbose=-1", null, lightgbmlib.voidpp_handle());
    }

    private final int num_instances;
    private final int num_features;
    private final String[] feature_names;
    private final SWIGTYPE_p_float swig_features;
    private final SWIGTYPE_p_float swig_labels;
    private final SWIGTYPE_p_float swig_weights;
    private final boolean has_labels;
    private final boolean has_weights;

    private final SWIGTYPE_p_p_void swig_out_data = lightgbmlib.voidpp_handle();
    private SWIGTYPE_p_void swig_data;

    public Dataset(float[][] data, String[] feature_names) {
        this(data, feature_names, null, null);
    }

    public Dataset(float[][] data, String[] feature_names, float[] labels) {
        this(data, feature_names, labels, null);
    }

    public Dataset(float[][] data, String[] feature_names, float[] labels, float[] weights) {
        if (data == null || data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("Dataset creation failed: empty array.");
        }

        if (feature_names == null || data[0].length != feature_names.length) {
            throw new IllegalArgumentException("Incorrect specifications for feature names");
        }
        this.has_labels = labels != null;
        this.has_weights = weights != null;

        if (has_labels && data.length != labels.length) {
            throw new IllegalArgumentException(
                    "Mismatched data size: #labels = " + labels.length + ", #instances = " + data.length);
        }

        if (has_weights && data.length != weights.length) {
            throw new IllegalArgumentException(
                    "Mismatched data size: #weights = " + weights.length + ", #instances = " + data.length);
        }

        this.num_instances = data.length;
        this.num_features = data[0].length;
        this.feature_names = feature_names.clone();
        this.swig_features = lightgbmlib.new_floatArray(num_instances * num_features);
        this.swig_labels = lightgbmlib.new_floatArray(num_instances);
        this.swig_weights = lightgbmlib.new_floatArray(num_instances);

        copyDataToSwigArray(data, labels, weights);
    }

    public int getNumOfInstances() {
        return num_instances;
    }

    public int getNumOfFeatures() {
        return num_features;
    }

    public SWIGTYPE_p_void getDataHandle(String params, boolean include_labels) {
        if (include_labels) {
            Utils.check(lightgbmlib.LGBM_DatasetCreateFromMat(lightgbmlib.float_to_voidp_ptr(swig_features),
                    lightgbmlibConstants.C_API_DTYPE_FLOAT32, num_instances, num_features, 1, params, null,
                    swig_out_data), "Dataset creation failed.");
            swig_data = lightgbmlib.voidpp_value(swig_out_data);

            setLabels(swig_data);

            return swig_data;
        } else {
            return lightgbmlib.float_to_voidp_ptr(swig_features);
        }
    }

    public void close() {
        lightgbmlib.delete_floatArray(swig_features);
        lightgbmlib.delete_floatArray(swig_labels);
        lightgbmlib.delete_voidpp(swig_out_data);
        lightgbmlib.LGBM_DatasetFree(swig_data);
    }

    private void setLabels(SWIGTYPE_p_void swig_data) {
        if (has_labels) {
            Utils.check(
                    lightgbmlib.LGBM_DatasetSetField(swig_data, "label", lightgbmlib.float_to_voidp_ptr(swig_labels),
                            num_instances, lightgbmlibConstants.C_API_DTYPE_FLOAT32),
                    "Failed to assign label field");

            if (has_weights) {
                Utils.check(lightgbmlib.LGBM_DatasetSetField(swig_data, "weight",
                        lightgbmlib.float_to_voidp_ptr(swig_weights), num_instances,
                        lightgbmlibConstants.C_API_DTYPE_FLOAT32), "Failed to assign weight field");

            }
        }

        Utils.check(lightgbmlib.LGBM_DatasetSetFeatureNames(swig_data, feature_names, num_features),
                "Failed to set feature names");

    }

    private void copyDataToSwigArray(float[][] data, float[] labels, float[] weights) {
        int ptr = 0;

        for (int i = 0; i < num_instances; i++) {
            for (int col_index = 0; col_index < num_features; ++col_index) {
                lightgbmlib.floatArray_setitem(swig_features, ptr++, data[i][col_index]);
            }

            if (labels != null) {
                lightgbmlib.floatArray_setitem(swig_labels, i, labels[i]);
            }

            if (weights != null) {
                lightgbmlib.floatArray_setitem(swig_weights, i, weights[i]);
            }
        }
    }

}