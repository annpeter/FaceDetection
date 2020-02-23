package cn.annpeter.cv.facedetection.tensorflow.tflite;


import java.util.LinkedList;
import java.util.List;

import javaslang.collection.Stream;

public class GenerateAnchorsUtils {

    public static float[][] generateAnchors(float[][] featureMapSizes, float[][] anchorSizes, float[][] anchorRatios) {
        int size = 0;

        List<float[][]> list = new LinkedList<>();
        for (int idx = 0; idx < featureMapSizes.length; idx++) {
            float[] featureSize = featureMapSizes[idx];

            Float[] cx = Stream.ofAll(linespace(0, featureSize[0] - 1, (int) (featureSize[0]))).map(item -> (item + 0.5F) / featureSize[0]).toJavaArray(Float.class);
            Float[] cy = Stream.ofAll(linespace(0, featureSize[1] - 1, (int) (featureSize[1]))).map(item -> (item + 0.5F) / featureSize[1]).toJavaArray(Float.class);

            float[][] cxGrid = meshgrid(cx, cy);
            float[][] cyGrid = reverse(cxGrid);

            int numAnchors = anchorSizes[idx].length + anchorRatios[idx].length - 1;
            float[][][] centerTiled = concat(cxGrid, cyGrid, 4 * numAnchors);


            List<Double> anchorWidthHeightList = new LinkedList<>();
            float[] anchorSize = anchorSizes[idx];
            for (int i = 0; i < anchorSize.length; i++) {
                float scale = anchorSize[i];
                float ratio = anchorRatios[idx][0];
                float width = (float) (scale * Math.sqrt(ratio));
                float height = (float)(scale / Math.sqrt(ratio));

                anchorWidthHeightList.add(-width / 2.0);
                anchorWidthHeightList.add(-height / 2.0);
                anchorWidthHeightList.add(width / 2.0);
                anchorWidthHeightList.add(height / 2.0);
            }


            float[] anchorRatio = anchorRatios[idx];
            for (int i = 1; i < anchorRatio.length; i++) {
                float ratio = anchorRatio[i];
                float s1 = anchorSize[0];

                float width = (float)(s1 * Math.sqrt(ratio));
                float height = (float)(s1 / Math.sqrt(ratio));

                anchorWidthHeightList.add(-width / 2.0);
                anchorWidthHeightList.add(-height / 2.0);
                anchorWidthHeightList.add(width / 2.0);
                anchorWidthHeightList.add(height / 2.0);
            }

            Double[] anchorWidthHeights = new Double[anchorWidthHeightList.size()];
            anchorWidthHeightList.toArray(anchorWidthHeights);

            add(centerTiled, anchorWidthHeights);
            float[][] bboxCoordsReshape = reshap(centerTiled);

            size += bboxCoordsReshape.length;
            list.add(bboxCoordsReshape);

        }

        return addTotal(list, size);
    }


    private static float[][] addTotal(List<float[][]> list, int size){
        float[][] result = new float[size][4];

        int index = 0;
        for (float[][] array: list){
            for (float[] line: array) {
                result[index++] = line;
            }
        }
        return result;
    }

    // 修改矩阵形状
    private static float[][] reshap(float[][][] centerTiled) {
        int size = centerTiled[0][0].length * centerTiled[0].length * centerTiled.length;

        float[][] result = new float[size / 4][4];

        int m = 0;
        int n = 0;

        for (int i = 0; i < centerTiled.length; i++) {
            for (int j = 0; j < centerTiled[0].length; j++) {
                for (int k = 0; k < centerTiled[0][0].length; k++) {
                    result[m][n] = centerTiled[i][j][k];

                    if (++n % 4 == 0) {
                        n = 0;
                        m++;
                    }
                }
            }
        }

        return result;
    }

    private static void add(float[][][] centerTiled, Double[] anchorWidthHeights) {
        for (int i = 0; i < centerTiled.length; i++) {
            for (int j = 0; j < centerTiled[0].length; j++) {
                for (int k = 0; k < centerTiled[0][0].length; k++) {
                    centerTiled[i][j][k] += anchorWidthHeights[k];
                }
            }
        }
    }

    private static float[][][] concat(float[][] cxGridExpend, float[][] cyGridExpend, int n) {
        float[][][] result = new float[cxGridExpend[0].length][cxGridExpend.length][n];

        for (int i = 0; i < cxGridExpend[0].length; i++) {
            for (int j = 0; j < cxGridExpend.length; j++) {
                float data1 = cxGridExpend[i][j];
                float data2 = cyGridExpend[i][j];
                for (int k = 0; k < n; k += 2) {
                    result[i][j][k] = data1;
                    result[i][j][k + 1] = data2;
                }
            }
        }
        return result;
    }


    // 坐标矩阵
    private static float[][] meshgrid(Float[] cx, Float[] cy) {
        float[][] result = new float[cx.length][cy.length];

        for (int j = 0; j < cy.length; j++) {
            for (int i = 0; i < cx.length; i++) {
                result[j][i] = cx[i];
            }
        }

        return result;
    }

    // 将矩阵转置
    public static float[][] reverse(float temp[][]) {
        float[][] result = new float[temp.length][temp[0].length];
        for (int i = 0; i < temp.length; i++) {
            for (int j = 0; j < temp[i].length; j++) {
                float k = temp[i][j];
                result[j][i] = k;
            }
        }

        return result;
    }

    // 等差数列
    private static float[] linespace(float start, float end, int count) {
        float space = (count-1) / (end - start);

        float[] result = new float[count];

        float data = start;
        for (int i = 0; i < count; i++) {
            result[i] = data;
            data += space;
        }
        return result;
    }
}
