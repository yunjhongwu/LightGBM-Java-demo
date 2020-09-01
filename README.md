LightGBM-Java-demo
==================
A simple example of [LightGBM](https://github.com/microsoft/LightGBM) Java Wrapper

Users may need to [build Java Wrapper](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#build-java-wrapper) by themselves instead of using pre-built files in the repo. The example includes the following LightGBM tasks in Java:
 * Create datasets from float arrays;
 * Initialize models;
 * Read models from string;
 * Train boosters;
 * Generate predictions from boosters;
 * Output models as string.

The code has been tested with LightGBM 3.0.0, and the jar of com.microsoft.ml.lightgbm is built manually as no official jar has been released at this moment.
