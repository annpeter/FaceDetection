package cn.annpeter.cv.facedetection.tensorflow.tflite;


import android.util.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class MnsUtils {

    static float confThresh = 0.5F;
    static float iouThresh = 0.4F;

    public static int[] singleClassNonMaxSuppression(float[][] bboxs, float[] confidences) {
        Integer[] confKeepIdx = getKeepIndex(confidences);

        confidences = getConfidences(confidences, confKeepIdx);

        float[][] bboxes = getIndex(bboxs, confKeepIdx);

        float[] xmin = getData3(bboxes, 0);
        float[] ymin = getData3(bboxes, 1);
        float[] xmax = getData3(bboxes, 2);
        float[] ymax = getData3(bboxes, 3);

        float[] area = area(bboxes);

        List<Integer> idxs = getConfIndex(confidences);

        List<Integer> pick = new LinkedList<>();
        while (idxs.size() > 0) {
            int last = idxs.size() - 1;
            int i = idxs.get(last);

            pick.add(i);

            float[] overlapXmin = getMaxinum(xmin[i], getData(xmin, idxs, last), true);
            float[] overlapYmin = getMaxinum(ymin[i], getData(ymin, idxs, last), true);
            float[] overlapXman = getMaxinum(xmax[i], getData(xmax, idxs, last), false);
            float[] overlapYman = getMaxinum(ymax[i], getData(ymax, idxs, last), false);

            float[] overlapW = getData1(overlapXman, overlapXmin);
            float[] overlapH = getData1(overlapYman, overlapYmin);

            float[] overlapArea = getOverlapArea(overlapW, overlapH);

            float[] overlapRatio = getOverlapRatio(overlapArea, area, i, last, idxs);

            List<Integer> needDelete = getNeedDelete(last, overlapRatio);

            idxs.removeAll(needDelete);
        }

        int[] res = getRes(pick, confKeepIdx);
        return res;
    }

    private static int[] getRes(List<Integer> pick, Integer[] confKeepIdx) {
        int[] res = new int[pick.size()];
        for (int i = 0; i < pick.size(); i++) {
            res[i] = confKeepIdx[pick.get(i)];
        }
        return res;
    }

    private static List<Integer> getNeedDelete(int last, float[] overlapRatio) {
        List<Integer> list = new LinkedList<>();
        list.add(last);
        for (int i = 0; i < overlapRatio.length; i++) {
            if (overlapRatio[i] > iouThresh) {
                list.add(i);
            }
        }
        return list;
    }

    private static float[] getData3(float[][] bboxes, int index) {
        float[] res = new float[bboxes.length];

        for (int i = 0; i < res.length; i++) {
            res[i] = bboxes[i][index];
        }
        return res;
    }

    private static float[] getConfidences(float[] confidences, Integer[] confKeepIdx) {
        float[] res = new float[confKeepIdx.length];
        for (int i = 0; i < confKeepIdx.length; i++) {
            res[i] = confidences[confKeepIdx[i]];
        }
        return res;
    }

    private static float[] getOverlapRatio(float[] overlapArea, float[] area, int index, int last, List<Integer> idxs) {
        float[] res = new float[last];

        for (int i = 0; i < last; i++) {
            res[i] = overlapArea[i] / (area[idxs.get(i)] + area[index] - overlapArea[i]);
        }

        return res;
    }

    private static float[] getOverlapArea(float[] overlapW, float[] overlapH) {
        float[] res = new float[overlapW.length];

        for (int i = 0; i < overlapW.length; i++) {
            res[i] = overlapW[i] * overlapH[i];
        }
        return res;
    }

    private static float[] getData1(float[] overlapMax, float[] overlapMin) {
        float[] res = new float[overlapMax.length];

        for (int i = 0; i < overlapMax.length; i++) {
            float v = overlapMax[i] - overlapMin[i];
            res[i] = v > 0 ? v : 0;
        }
        return res;
    }

    private static float[] getData(float[] data, List<Integer> idxs, int last) {
        float[] res = new float[last];

        for (int i = 0; i < last; i++) {
            res[i] = data[idxs.get(i)];
        }
        return res;
    }


    private static float[] getMaxinum(float data, float[] range, boolean getMax) {
        float[] res = new float[range.length];

        for (int i = 0; i < range.length; i++) {
            if (getMax) {
                res[i] = range[i] > data ? range[i] : data;
            } else {
                res[i] = range[i] < data ? range[i] : data;
            }
        }
        return res;
    }

    private static List<Integer> getConfIndex(float[] confidences) {
        List<Pair<Float, Integer>> list = new ArrayList<>();

        for (int i = 0; i < confidences.length; i++) {
            list.add(Pair.create(confidences[i], i));
        }

        Collections.sort(list, (o1, o2) -> (int) (o1.first - o2.first));

        List<Integer> idxs = new ArrayList<>(list.size());
        for (Pair<Float, Integer> item : list) {
            idxs.add(item.second);
        }
        return idxs;
    }


    private static float[] area(float[][] bboxes) {
        float[] res = new float[bboxes.length];

        for (int i = 0; i < res.length; i++) {
            res[i] = (float) ((bboxes[i][2] - bboxes[i][0] + Math.pow(10, -3)) * (bboxes[i][3] - bboxes[i][1] + Math.pow(10, -3)));
        }

        return res;
    }

    private static Integer[] getKeepIndex(float[] confidences) {
        List<Integer> list = new LinkedList<>();
        for (int i = 0; i < confidences.length; i++) {
            if (confidences[i] > confThresh) {
                list.add(i);
            }
        }
        Integer[] arr = new Integer[list.size()];
        return list.toArray(arr);
    }

    private static float[][] getIndex(float[][] bbox, Integer[] confKeepIdx) {
        float[][] bboxes = new float[confKeepIdx.length][4];
        for (int i = 0; i < confKeepIdx.length; i++) {
            float[] bboxT = bbox[confKeepIdx[i]];
            bboxes[i] = bboxT;
        }
        return bboxes;
    }

}
