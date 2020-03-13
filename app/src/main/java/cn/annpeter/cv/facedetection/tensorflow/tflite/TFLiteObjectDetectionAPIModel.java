package cn.annpeter.cv.facedetection.tensorflow.tflite;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat4;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.samples.facedetect.DetectResult;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import static org.opencv.core.Core.NORM_MINMAX;


/**
 * Wrapper for frozen detection models trained using the Tensorflow Object Detection API:
 * github.com/tensorflow/models/tree/master/research/object_detection
 */
public class TFLiteObjectDetectionAPIModel implements Classifier {

    private static final int NUM_THREADS = 2;
    private Interpreter tfLite;

    private ByteBuffer imgData;

    private int inputSize = 260;

    private static final int NUM_DETECTIONS = 5972;

    private float[][][] outputLocations;
    private float[][][] outputClasses;

    private static float[][] featureMapSizes = new float[][]{{33, 33}, {17, 17}, {9, 9}, {5, 5}, {3, 3}};
    private static float[][] anchorSizes = new float[][]{{0.04F, 0.056F}, {0.08F, 0.11F}, {0.16F, 0.22F}, {0.32F, 0.45F}, {0.64F, 0.72F}};
    private static float[][] anchorRatios = new float[][]{{1, 0.62F, 0.42F}, {1, 0.62F, 0.42F}, {1, 0.62F, 0.42F}, {1, 0.62F, 0.42F}, {1, 0.62F, 0.42F}};

    private static float[][][] anchorsExp;

    private TFLiteObjectDetectionAPIModel() {
    }

    /**
     * Memory-map the model file in Assets.
     */
    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     */
    public static Classifier create(final AssetManager assetManager, final String modelFilename) {
        final TFLiteObjectDetectionAPIModel d = new TFLiteObjectDetectionAPIModel();
        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
            d.tfLite.setNumThreads(NUM_THREADS);

            int numBytesPerChannel = 4;

            d.imgData = ByteBuffer.allocateDirect(d.inputSize * d.inputSize * 3 * numBytesPerChannel);
            d.imgData.order(ByteOrder.nativeOrder());

            d.outputLocations = new float[1][NUM_DETECTIONS][4];
            d.outputClasses = new float[1][NUM_DETECTIONS][2];

            float[][] doubles = GenerateAnchorsUtils.generateAnchors(featureMapSizes, anchorSizes, anchorRatios);
            anchorsExp = new float[1][doubles.length][1];
            anchorsExp[0] = doubles;

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
        return d;
    }


    private MatOfFloat4 convert(Mat gray) {
        Mat resizeOut = new Mat();
        Imgproc.resize(gray, resizeOut, new Size(260, 260));

        MatOfFloat4 resizeFloatOut = new MatOfFloat4();
        resizeOut.convertTo(resizeFloatOut, CvType.CV_32F);

        resizeOut.release();

        MatOfFloat4 normalizeOut = new MatOfFloat4();
        Core.normalize(resizeFloatOut, normalizeOut, 0.0, 1.0, NORM_MINMAX);
        resizeFloatOut.release();
        return normalizeOut;
    }

    @RequiresApi(api = Build.VERSION_CODES.JELLY_BEAN_MR2)
    public DetectResult detectObject(Mat mRgba) {
        imgData.rewind();

        MatOfFloat4 colorOut = null;
        try {
            colorOut = convert(mRgba);
            Size size = mRgba.size();
            int width = (int) size.width;
            int height = (int) size.height;

            for (int i = 0; i < inputSize; ++i) {
                for (int j = 0; j < inputSize; ++j) {
                    double[] data = colorOut.get(i, j);
                    imgData.putFloat((float) data[2]);
                    imgData.putFloat((float) data[1]);
                    imgData.putFloat((float) data[0]);
                }
            }
            colorOut.release();

            Object[] inputArray = {imgData};
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, outputLocations);
            outputMap.put(1, outputClasses);

            long l = System.currentTimeMillis();
            tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

            long l2 = System.currentTimeMillis();

            float[][] bbox = AnchorDecodeUtils.decodeBbox(anchorsExp, outputLocations);

            float[] bboxMaxScores = ResultHandUtils.max(outputClasses);

            int[] ints = MnsUtils.singleClassNonMaxSuppression(bbox, bboxMaxScores);

            float  confidence = 1;
            Rect[] rects = new Rect[ints.length];
            for (int i = 0; i < 1; i++) {
                float[] bbox1 = bbox[ints[i]];
                confidence = bboxMaxScores[i];

                int xmin = Math.max(0, (int) (bbox1[0] * width));
                int ymin = Math.max(0, (int) (bbox1[1] * height));
                int xmax = Math.min((int) (bbox1[2] * width), width);
                int ymax = Math.min((int) (bbox1[3] * height), height);

                rects[i] = new Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            }

            Log.d("FACE", "预测耗时" + (l2 - l) + "   总耗时：" + (System.currentTimeMillis() - l));

            return new DetectResult(rects,  (int)(l2 - l),  (int) (System.currentTimeMillis() - l), confidence);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (colorOut != null) {
                colorOut.release();
            }
        }
        return null;
    }
}
