package cn.annpeter.cv.facedetection.tensorflow.tflite;


public class AnchorDecodeUtils {

    static float[] variances = new float[]{0.1F, 0.1F, 0.2F, 0.2F};

    public static float[][] decodeBbox(float[][][] anchors, float[][][] outputLocations) {
        float[][][] anchorCentersX = add(anchors, 0, 2, true, (data -> data / 2));
        float[][][] anchorCentersY = add(anchors, 1, 3, true, (data -> data / 2));


        float[][][] anchorsW = add(anchors, 2, 0, false, null);
        float[][][] anchorsH = add(anchors, 3, 1, false, null);

        product(outputLocations, variances);

        float[][][] predictCenterX = product1(outputLocations, 0, anchorsW, anchorCentersX);
        float[][][] predictCenterY = product1(outputLocations, 1, anchorsH, anchorCentersY);

        float[][][] predictW = product2(outputLocations, 2, anchorsW);
        float[][][] predictH = product2(outputLocations, 3, anchorsH);

        float[][][] predictXmin = product3(predictCenterX, predictW, false);
        float[][][] predictYmin = product3(predictCenterY, predictH, false);
        float[][][] predictXman = product3(predictCenterX, predictW, true);
        float[][][] predictYman = product3(predictCenterY, predictH, true);

        float[][] concat = concat(predictXmin, predictYmin, predictXman, predictYman);
        return concat;
    }


    private static float[][] concat(float[][][] predictXmin, float[][][] predictYmin, float[][][] predictXman, float[][][] predictYman) {
        float[][] res = new float[predictXmin[0].length][4];

        for (int i = 0; i < predictXmin[0].length; i++) {
            res[i][0] = predictXmin[0][i][0];
            res[i][1] = predictYmin[0][i][0];
            res[i][2] = predictXman[0][i][0];
            res[i][3] = predictYman[0][i][0];
        }

        return res;
    }

    private static float[][][] product3(float[][][] anchorCenters, float[][][] predict, boolean add) {
        float[][][] res = new float[1][anchorCenters[0].length][1];
        for (int i = 0; i < anchorCenters[0].length; i++) {
            res[0][i][0] = add ? anchorCenters[0][i][0] + predict[0][i][0] / 2 : anchorCenters[0][i][0] - predict[0][i][0] / 2;
        }
        return res;
    }

    private static float[][][] product2(float[][][] outputLocations, int index, float[][][] anchorsW) {
        float[][][] res = new float[1][outputLocations[0].length][1];

        for (int i = 0; i < outputLocations[0].length; i++) {
            res[0][i][0] = (float) Math.exp(outputLocations[0][i][index]) * anchorsW[0][i][0];
        }

        return res;
    }


    private static float[][][] product1(float[][][] outputLocations, int index, float[][][] anchorsW, float[][][] anchorCentersX) {
        float[][][] res = new float[1][outputLocations[0].length][1];

        for (int i = 0; i < outputLocations[0].length; i++) {
            res[0][i][0] = outputLocations[0][i][index] * anchorsW[0][i][0] + anchorCentersX[0][i][0];
        }
        return res;
    }


    private static void product(float[][][] outputLocations, float[] variances) {

        for (int i = 0; i < outputLocations.length; i++) {
            for (int j = 0; j < outputLocations[0].length; j++) {
                for (int k = 0; k < outputLocations[0][0].length; k++) {
                    outputLocations[i][j][k] = outputLocations[i][j][k] * variances[k];
                }
            }
        }
    }


    private static float[][][] add(float[][][] anchors, int m, int n, boolean add, ResultHandler resultHandler) {
        float[][][] result = new float[1][anchors[0].length][1];

        for (int i = 0; i < anchors.length; i++) {
            for (int j = 0; j < anchors[0].length; j++) {
                float data = add ? anchors[i][j][m] + anchors[i][j][n] : anchors[i][j][m] - anchors[i][j][n];

                if (resultHandler != null) {
                    result[0][j][0] = resultHandler.run(data);
                } else {
                    result[0][j][0] = data;
                }
            }
        }

        return result;
    }

    interface ResultHandler {
        float run(float data);
    }
}
