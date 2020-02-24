package org.opencv.samples.facedetect;

import org.opencv.core.Rect;

import lombok.AllArgsConstructor;
import lombok.Getter;

@AllArgsConstructor
@Getter
public class DetectResult {
    private Rect[] facesArray;
    private int predictTime;
    private int totalTime;
}
