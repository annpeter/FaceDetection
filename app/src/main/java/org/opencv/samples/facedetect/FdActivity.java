package org.opencv.samples.facedetect;

import android.content.res.AssetManager;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.InputStream;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import cn.annpeter.cv.facedetection.tensorflow.tflite.Classifier;
import cn.annpeter.cv.facedetection.tensorflow.tflite.TFLiteObjectDetectionAPIModel;

import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

public class FdActivity extends CameraActivity implements CvCameraViewListener2, View.OnTouchListener {

    private static final String TAG = "OCVSample::Activity";

    private Classifier classifier;
    private AtomicInteger picktureIndex = new AtomicInteger(1);
    private Mat pickture;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    AssetManager assetManager = getAssets();
                    classifier = TFLiteObjectDetectionAPIModel.create(assetManager, "face_mask_detection.tflite");

                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public FdActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCameraIndex(1);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setClickable(true);
        mOpenCvCameraView.setOnTouchListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
    }

    public void onCameraViewStopped() {
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat mRgba = inputFrame.rgba();

        try (InputStream is = getAssets().open("test_" + picktureIndex.get() % 8 + ".jpg")) {
            pickture = Utils.loadResource(is, IMREAD_COLOR);

            mRgba.release();

            mRgba = pickture;

            DetectResult detectResult = classifier.detectObject(mRgba);
            Rect[] facesArray = detectResult.getFacesArray();

            for (int i = 0; i < facesArray.length; i++) {
                Scalar scalar;
                String tag;
                if (picktureIndex.get() % 8 < 4) {
                    tag = "FACE: ";
                    scalar = new Scalar(0, 255, 0, 0);
                } else {
                    tag = "MASK: ";
                    scalar = new Scalar(0, 0, 255, 0);
                }

                Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), scalar, 3);
                Imgproc.putText(mRgba, tag + "PredictTime:" + detectResult.getPredictTime()
                                + ", TotalTime:" + detectResult.getTotalTime() + ", Confidence=" + detectResult.getConfidence(),
                        new Point(20, 100), FONT_HERSHEY_SIMPLEX, 2, scalar, 5);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Core.flip(mRgba, mRgba, 1);
        return mRgba;
    }

    private long mLastClickTime = 0;
    public static final long TIME_INTERVAL = 1000L;

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        long nowTime = System.currentTimeMillis();
        if (nowTime - mLastClickTime > TIME_INTERVAL) {
            // do something
            mLastClickTime = nowTime;
            picktureIndex.incrementAndGet();
        }
        return false;
    }
}
