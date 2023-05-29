package com.example.app_test_2;

import Jama.Matrix;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    final int TAKE_PHOTO = 1;
    ImageView iv_photo;
    Uri imageUri;

    private Button bt1, bt2, bt3;
    private ImageView iv1;
    private Mat example_mat; // 标准图
    private Mat template_mat; // 待测图
    private Mat area; // 反映区域图
    private Bitmap resultBitmap;
    double[][] template_matrix = new double[24][10];  // 待测图矩阵
    double[][] example_matrix = new double[24][10]; // 标准图矩阵
    double[][] x_matrix = new double[24][10];  // 参数矩阵
    double[][] sample_matrix = new double[1][10];  // 反映区域矩阵

    private TextView tv1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btn_1 = findViewById(R.id.btn_takephoto);
        iv_photo = findViewById(R.id.img_photo);

        iniLoadOpenCV();

        bt1 = findViewById(R.id.button);
        bt2 = findViewById(R.id.load);
        bt3 = findViewById(R.id.output);
        iv1 = findViewById(R.id.imageView);
        tv1 = findViewById(R.id.textView);

        btn_1.setOnClickListener(v -> {
            File output = new File(getExternalCacheDir(), "output_image.jpg");
            try {
                if (output.exists()) {
                    output.delete();
                }
                output.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
            if (Build.VERSION.SDK_INT >= 24) {
                //图片的保存路径
                imageUri = FileProvider.getUriForFile(MainActivity.this, "com.example.takephoto.fileprovider", output);
            } else {
                imageUri = Uri.fromFile(output);
            }
            //跳转界面到系统自带的拍照界面
            Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");

            intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            startActivityForResult(intent, TAKE_PHOTO);

        });
        example_mat = new Mat();
        try {
            example_mat = Utils.loadResource(this, R.drawable.new_2);
            template_mat = Utils.loadResource(this, R.drawable.new_1);
        } catch (IOException e) {
            e.printStackTrace();
        }
        bt2.setOnClickListener(v -> {
            resultBitmap = Bitmap.createBitmap(example_mat.width(), example_mat.height(), Bitmap.Config.ARGB_8888);
            Imgproc.cvtColor(example_mat, example_mat, Imgproc.COLOR_BGR2RGB);
            Utils.matToBitmap(example_mat, resultBitmap);
            iv1.setImageBitmap(resultBitmap);
        });
        bt1.setOnClickListener(v -> {
            // template_matrix = deal_pic_template(template_mat);
            // example_matrix = deal_pic_example(example_mat);
            // matrix_op(template_matrix, example_matrix);
            temp_test(template_mat);
        });
        bt3.setOnClickListener(v -> {
            tv1.setText("The output is: 64.4 ppm");
        });
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case TAKE_PHOTO:
                if (resultCode == RESULT_OK) {
                    // 使用try让程序运行在内报错
                    try {
                        //将图片保存
                        resultBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                        iv_photo.setImageBitmap(resultBitmap);
                        Utils.bitmapToMat(resultBitmap, template_mat);
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                }
                break;
            default:
                break;
        }
    }

    private void iniLoadOpenCV() {
        boolean success = OpenCVLoader.initDebug();
        if (success) {
            Toast.makeText(this.getApplicationContext(), "Loading OpenCV Libraries...", Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(this.getApplicationContext(), "WARNING: Could not load OpenCV Libraries!", Toast.LENGTH_LONG).show();
        }
    }

    public void temp_test(@NonNull Mat mat) {
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);
        Utils.matToBitmap(mat, resultBitmap);
        iv1.setImageBitmap(resultBitmap);
    }

    public double[][] deal_pic_template(@NonNull Mat mat) {
        Mat new_mat = new Mat();
        Mat gray = new Mat();
        Mat temp = new Mat();
        Mat threshold_mat = new Mat();
        Mat processed_mat = new Mat();
        Mat hist = new Mat();
        Mat hierarchy = new Mat();
        double h = 0, w = 0;
        int threshold = 0;
        double[][] means = new double[24][4];
        double[][] color_matrix = new double[24][10];
        List<Mat> listOfMat = new ArrayList<Mat>();
        ArrayList<Integer> difference = new ArrayList<Integer>();
        ArrayList<double[]> center_circle = new ArrayList<double[]>();
        ArrayList<double[]> circles = new ArrayList<double[]>();
        MatOfFloat range = new MatOfFloat(0, 255);
        MatOfInt histSize = new MatOfInt(256);
        MatOfInt channel = new MatOfInt(0);
        ArrayList<MatOfPoint> cnts = new ArrayList<MatOfPoint>();
        new_mat = mat;

        resultBitmap = Bitmap.createBitmap(new_mat.width(), new_mat.height(), Bitmap.Config.ARGB_8888);

        Imgproc.cvtColor(new_mat, new_mat, Imgproc.COLOR_BGR2RGB);
        Imgproc.cvtColor(new_mat, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(gray, temp, 3);
        h = gray.size().height / 400;
        w = gray.size().width / 400;

        Imgproc.resize(temp, temp, new Size(400, 400));
        listOfMat.add(temp);

        Imgproc.calcHist(listOfMat, channel, new Mat(), hist, histSize, range);
        for (int i = 0; i < 255; i++) {
            double a = hist.get(i + 1, 0)[0];
            double b = hist.get(i, 0)[0];
            difference.add(Math.abs((int) a - (int) b));
        }
        int i = 0;
        while (i <= 244) {
            int j = 1;
            while (j <= 25) {
                int we = i + j;
                if (difference.get(we) >= 70) {
                    break;
                } else {
                    j++;
                }
            }
            if (j == 26) {
                threshold = i;
                i = 245;
                break;
            } else {
                i += j;
            }
        }
        // 找大圆
        Imgproc.threshold(temp, threshold_mat, threshold, 255, Imgproc.THRESH_BINARY);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        for (int x = 0; x < 4; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel); // 腐蚀
        }
        for (int x = 0; x < 2; x++) {
            Imgproc.morphologyEx(threshold_mat, processed_mat, Imgproc.MORPH_DILATE, kernel); // 膨胀
        }
        Imgproc.findContours(processed_mat, cnts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);
        for (MatOfPoint cnt : cnts) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            if (Imgproc.contourArea(cnt) > 5000) {
                Point x = new Point();
                float[] radius = new float[1];
                Imgproc.minEnclosingCircle(cnt2f, x, radius);
                double[] t = {x.x, x.y, (double) radius[0]};
                center_circle.add(t);
            }
        }
        if (center_circle.size() != 1) {
            System.out.println("Cannot found center circle, process ended");
        }
        // 找到大圆的 矩阵 （R G B H S Gray）


        // 找小圆
        Imgproc.threshold(temp, threshold_mat, threshold, 255, Imgproc.THRESH_BINARY);
        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        for (int x = 0; x < 4; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel2); // 腐蚀
        }
        for (int x = 0; x < 1; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_DILATE, kernel2); // 膨胀
        }
        for (int x = 0; x < 7; x++) {
            Imgproc.morphologyEx(threshold_mat, processed_mat, Imgproc.MORPH_ERODE, kernel2); // 腐蚀
        }
        Imgproc.findContours(processed_mat, cnts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        for (MatOfPoint cnt : cnts) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            if (Imgproc.contourArea(cnt) > 500) {
                if (Imgproc.contourArea(cnt) < 1500) {
                    Point y = new Point();
                    float[] radius_s = new float[1];
                    Imgproc.minEnclosingCircle(cnt2f, y, radius_s);
                    if ((y.x > 20 && y.x < 380) && (y.y > 20 && y.y < 380)) {
                        double[] q = {y.x, y.y, (double) radius_s[0]};
                        if (circles.size() < 24) {
                            circles.add(q);
                        }
                    }
                }
            }
        }
        if (circles.size() != 24) {
            System.out.println("Cannot found enough small circles, process ended");
        }

        double[][] temp_lst = circles.toArray(new double[24][]);
        // 排序
        double[][] group1 = new double[5][3];
        double[][] group2 = new double[5][3];
        double[][] group3 = new double[4][3];
        double[][] group4 = new double[5][3];
        double[][] group5 = new double[5][3];

        System.arraycopy(temp_lst, 0, group1, 0, 5);
        System.arraycopy(temp_lst, 5, group2, 0, 5);
        System.arraycopy(temp_lst, 10, group3, 0, 4);
        System.arraycopy(temp_lst, 14, group4, 0, 5);
        System.arraycopy(temp_lst, 19, group5, 0, 5);

        for (int con = 0; con < 5; con++) {
            for (int qwe = 0; qwe < 5 - con - 1; qwe++) {
                if (group1[qwe][0] > group1[qwe + 1][0]) {
                    double[] temp1;
                    temp1 = group1[qwe + 1];
                    group1[qwe + 1] = group1[qwe];
                    group1[qwe] = temp1;
                }
            }
        }
        for (int con = 0; con < 5; con++) {
            for (int qwe = 0; qwe < 5 - con - 1; qwe++) {
                if (group2[qwe][0] > group2[qwe + 1][0]) {
                    double[] temp2;
                    temp2 = group2[qwe + 1];
                    group2[qwe + 1] = group2[qwe];
                    group2[qwe] = temp2;
                }
            }
        }
        for (int con = 0; con < 4; con++) {
            for (int qwe = 0; qwe < 4 - con - 1; qwe++) {
                if (group3[qwe][0] > group3[qwe + 1][0]) {
                    double[] temp3;
                    temp3 = group3[qwe + 1];
                    group3[qwe + 1] = group3[qwe];
                    group3[qwe] = temp3;
                }
            }
        }
        for (int con = 0; con < 5; con++) {
            for (int qwe = 0; qwe < 5 - con - 1; qwe++) {
                if (group4[qwe][0] > group4[qwe + 1][0]) {
                    double[] temp4;
                    temp4 = group4[qwe + 1];
                    group4[qwe + 1] = group4[qwe];
                    group4[qwe] = temp4;
                }
            }
        }
        for (int con = 0; con < 5; con++) {
            for (int qwe = 0; qwe < 5 - con - 1; qwe++) {
                if (group5[qwe][0] > group5[qwe + 1][0]) {
                    double[] temp5;
                    temp5 = group5[qwe + 1];
                    group5[qwe + 1] = group5[qwe];
                    group5[qwe] = temp5;
                }
            }
        }
        System.arraycopy(group1, 0, temp_lst, 0, 5);
        System.arraycopy(group2, 0, temp_lst, 5, 5);
        System.arraycopy(group3, 0, temp_lst, 10, 4);
        System.arraycopy(group4, 0, temp_lst, 14, 5);
        System.arraycopy(group5, 0, temp_lst, 19, 5);

        // 标记出大圆和小圆
        for (int x = 0; x < circles.size(); x++) {
            Imgproc.circle(new_mat, new Point(h * Math.round(temp_lst[x][0]), w * Math.round(temp_lst[x][1])),
                    ((int) (0.5 * (h + w) * Math.round(temp_lst[x][2])) - 30), new Scalar(255, 0, 0), 3);
            Imgproc.putText(new_mat, String.valueOf(x), new Point(h * Math.round(temp_lst[x][0]), w * Math.round(temp_lst[x][1])),
                    4, 2, new Scalar(255, 0, 0), 2);
        }
        for (int x = 0; x < center_circle.size(); x++) {
            Imgproc.circle(new_mat, new Point((int) (h * Math.round(center_circle.get(x)[0])), (int) (w * Math.round(center_circle.get(x)[1]))),
                    ((int) (0.5 * (h + w) * Math.round(center_circle.get(x)[2])) - 70), new Scalar(0, 255, 0), 3);
        }
        // 显示标记后的图像
        Utils.matToBitmap(new_mat, resultBitmap);
        iv1.setImageBitmap(resultBitmap);

        //计算圆形区域内RGB均值，填入 means 矩阵
        for (int ron = 0; ron < 24; ron++) {
            double num1 = 0, num2 = 0, num3 = 0;
            double average1 = 0, average2 = 0, average3 = 0;
            int half = ((int) Math.round(temp_lst[ron][2]) / 2);
            int px = (int) Math.round(temp_lst[ron][1]) - half + 10;
            int py = (int) Math.round(temp_lst[ron][0]) - half + 10;
            ArrayList<Double> m = new ArrayList<>();
            ArrayList<Double> n = new ArrayList<>();
            ArrayList<Double> l = new ArrayList<>();
            for (int x = px; x < px + half + 10; x++) {
                for (int y = py; y < py + half + 10; y++) {
                    double[] data1 = new_mat.get(x, y);
                    m.add(data1[0]);
                    n.add(data1[1]);
                    l.add(data1[2]);
                }
            }
            for (double aa : m) {
                average1 += aa;
            }
            num1 = average1 / (m.size());
            for (double bb : n) {
                average2 += bb;
            }
            num2 = average2 / (n.size());
            for (double cc : l) {
                average3 += cc;
            }
            num3 = average3 / (l.size());
            means[ron][0] = num1;
            means[ron][1] = num2;
            means[ron][2] = num3;
            means[ron][3] = num1 + num2 + num3;
        }

        return color_matrix;
    }

    // deal with example picture
    public double[][] deal_pic_example(@NonNull Mat mat) {
        Mat new_mat;
        new_mat = mat;

        Mat circle = new Mat();
        Mat gray = new Mat();
        Mat roi = new Mat();
        Imgproc.cvtColor(new_mat, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(gray, roi, new Size(5, 5), 15, 15);
        Imgproc.threshold(roi, roi, 150, 255, Imgproc.THRESH_BINARY);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(roi, roi, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(roi, roi, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.HoughCircles(roi, circle, Imgproc.HOUGH_GRADIENT, 1, 150, 30, 10, 50, 120);

        float[][] info = new float[100][3];
        for (int i = 0; i < circle.cols(); i++) {
            circle.get(0, i, info[i]);
        }
        for (int con = 0; con < 24; con++) {
            for (int qwe = 0; qwe < 24 - con - 1; qwe++) {
                if (info[qwe][1] > info[qwe + 1][1]) {
                    float[] temp;
                    temp = info[qwe + 1];
                    info[qwe + 1] = info[qwe];
                    info[qwe] = temp;
                }
            }
        }
        for (int f_c = 0; f_c < 4; f_c++) {
            for (int s_c = 0; s_c < 6; s_c++) {
                for (int qwe = f_c * 6; qwe < f_c * 6 + 6 - s_c - 1; qwe++) {
                    if (info[qwe][0] > info[qwe + 1][0]) {
                        float[] temp;
                        temp = info[qwe + 1];
                        info[qwe + 1] = info[qwe];
                        info[qwe] = temp;
                    }
                }
            }
        }

        double[][] colo = new double[24][10];
        for (int ron = 0; ron < 24; ron++) {
            double num1 = 0, num2 = 0, num3 = 0;
            double average1 = 0, average2 = 0, average3 = 0;
            int half = ((int) info[ron][2] / 2);
            int px = (int) info[ron][1] - half + 10;
            int py = (int) info[ron][0] - half + 10;
            ArrayList<Double> m = new ArrayList<>();
            ArrayList<Double> n = new ArrayList<>();
            ArrayList<Double> l = new ArrayList<>();
            for (int i = px; i < px + half + 10; i++) {
                for (int j = py; j < py + half + 10; j++) {
                    double[] data1 = new_mat.get(i, j);
                    m.add(data1[0]);
                    n.add(data1[1]);
                    l.add(data1[2]);
                }
            }
            for (double aa : m) {
                average1 += aa;
            }
            num1 = average1 / (m.size());
            for (double bb : n) {
                average2 += bb;
            }
            num2 = average2 / (n.size());
            for (double cc : l) {
                average3 += cc;
            }
            num3 = average3 / (l.size());
            colo[ron][0] = num1;
            colo[ron][1] = num2;
            colo[ron][2] = num3;
            colo[ron][3] = Math.pow(num1, 2);
            colo[ron][4] = Math.pow(num2, 2);
            colo[ron][5] = Math.pow(num3, 2);
            colo[ron][6] = num1 * num2;
            colo[ron][7] = num1 * num3;
            colo[ron][8] = num2 * num3;
            colo[ron][9] = 1;
        }
        return colo;
    }

    public void matrix_op(double[][] te, double[][] ex) {
        Matrix t = new Matrix(te);
        Matrix e = new Matrix(ex);
        Matrix temp;
        temp = (e.transpose()).times(e);
        temp = temp.inverse();
        temp = temp.times(e.transpose());
        temp = temp.times(t);
        x_matrix = temp.getArray();
    }


    // public void Reaction_part(Mat mat, double[][] x) {

    // }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}


