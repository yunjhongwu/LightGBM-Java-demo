package test;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import main.base.Booster;
import main.base.Dataset;

public class Test {
    public static void main(String[] args) {
        // Generating sample data
        Random rnd = new Random();
        int size = 10247;
        int dim = 8;
        String[] feature_names = IntStream.range(0, dim).mapToObj(i -> "feature_" + i).toArray(String[]::new);
        float[][] data_train = new float[size][dim];
        float[][] data_test = new float[4][dim];
        float[] labels = new float[size];

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < dim; j++) {
                data_train[i][j] = rnd.nextFloat();
            }
            labels[i] = rnd.nextDouble() > 0.5 ? 1.0f : 0.0f;
        }

        for (int i = 0; i < data_test.length; i++) {
            for (int j = 0; j < dim; j++) {
                data_test[i][j] = rnd.nextFloat();
            }
        }

        Dataset dataset = new Dataset(data_train, feature_names, labels);

        // Run Booster methods
        Booster booster1 = new Booster(32, Map.of("objective", "binary", "num_class", 1, "force_col_wise", true));

        booster1.train(dataset);
        Booster booster2 = new Booster(booster1.getModelString());
        System.out.println(Arrays.toString(booster2.predict(new Dataset(data_test, feature_names))));
    }
}
