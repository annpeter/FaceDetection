package cn.annpeter.cv.facedetection.tensorflow.tflite;

public class ResultHandUtils {

    public static float[] max(float[][][] outputClasses){
        float[] res = new float[outputClasses[0].length];
        for (int i = 0; i < outputClasses[0].length; i++) {
            float[] outputClass = outputClasses[0][i];

            if(outputClass[0] > outputClass[1]){
                res[i] = outputClass[0];
            }else {
                res[i] = outputClass[1];
            }
        }

        return res;
    }

}
