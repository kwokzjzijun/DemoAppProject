package com.example.app_test_2;

import android.content.ContentResolver;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.ContactsContract;
import android.provider.MediaStore;
import android.util.Log;
import android.webkit.MimeTypeMap;
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
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import Jama.Matrix;

public class MainActivity extends AppCompatActivity {
    final int TAKE_PHOTO = 1;

    private static final int GALLERY_CODE = 19;
    ImageView iv_photo;
    Uri imageUri;

    private Button bt2, bt1, bt3, bt4;
    private ImageView iv1;
    private Mat example_mat; // standard graph
    private Mat template_mat; // Picture to be tested
    private Mat main_circle;
    private Bitmap resultBitmap;
    double[][] template_matrix = new double[24][3];
    double[][] example_matrix = new double[24][3];
    double[][] x_matrix = new double[3][3];  // parameter matrix

    private TextView tv1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button btn_1 = findViewById(R.id.btn_takephoto);
//        iv_photo = findViewById(R.id.img_photo);
        iv_photo = findViewById(R.id.imageView);
        iniLoadOpenCV();

        bt2 = findViewById(R.id.button);
        bt1 = findViewById(R.id.load);
        bt3 = findViewById(R.id.output);
        bt4 = findViewById(R.id.gallery);
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
                // Image saving path
                imageUri = FileProvider.getUriForFile(MainActivity.this, "com.example.takephoto.fileprovider", output);
            } else {
                imageUri = Uri.fromFile(output);
            }
            // Jump to the system’s own camera interface
            Intent intent = new Intent("android.media.action.IMAGE_CAPTURE");

            intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            startActivityForResult(intent, TAKE_PHOTO);

        });
        example_mat = new Mat();
        try {
            // may change the default example picture
            example_mat = Utils.loadResource(this, R.drawable.pic_0_4);
            template_mat = Utils.loadResource(this, R.drawable.pic_0_4);
        } catch (IOException e) {
            e.printStackTrace();
        }
        bt1.setOnClickListener(v -> {
            // the second button: load the picture
            resultBitmap = Bitmap.createBitmap(example_mat.width(), example_mat.height(), Bitmap.Config.ARGB_8888);
            Imgproc.cvtColor(example_mat, example_mat, Imgproc.COLOR_BGR2RGB);
            Utils.matToBitmap(example_mat, resultBitmap);
            iv1.setImageBitmap(resultBitmap);
        });
        //
        bt2.setOnClickListener(v -> {
            // the third button: execute the picture
            deal_pic_template(template_mat);
            deal_pic_example(example_mat);

            matrix_op(template_matrix, example_matrix);
        });
        bt4.setOnClickListener(v -> {
            Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(gallery, GALLERY_CODE);
        });
        bt3.setOnClickListener(v -> Reaction_part(main_circle, x_matrix));
        // the forth button: output the final value
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == TAKE_PHOTO) {
            if (resultCode == RESULT_OK) {
                // Use try to let the program run and report errors internally
                try {
                    // save the picture
                    resultBitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(imageUri));
                    iv_photo.setImageBitmap(resultBitmap);
                    Utils.bitmapToMat(resultBitmap, template_mat);
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
        }
        if (requestCode == GALLERY_CODE) {
            if (data == null) {
                return;
            } else {
                Uri uripic = data.getData();
                String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
                String imageFileName = "JPEG_" + timeStamp + "." + getFileExt(uripic);
                Log.d("tag", "onActivityResult: Gallery Image Uri:  " + imageFileName);
//                intent = PhotoUtils.startPhotoZoom(uripic, uri, 2000, 2000);
//                startActivityForResult(intent, CROP_CODE);
                resultBitmap = decodeUriAsBitmap(uripic);
                iv_photo.setImageBitmap(resultBitmap);
                Utils.bitmapToMat(resultBitmap, template_mat);
                Imgproc.cvtColor(template_mat, template_mat, Imgproc.COLOR_BGR2RGB);
            }
        }
    }
    private String getFileExt(Uri contentUri) {
        ContentResolver c = getContentResolver();
        MimeTypeMap mime = MimeTypeMap.getSingleton();
        return mime.getExtensionFromMimeType(c.getType(contentUri));
    }
    private Bitmap decodeUriAsBitmap(Uri uri) {
        Bitmap bitmap = null;
        try {
            bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
        return bitmap;
    }

    private void iniLoadOpenCV() {
        boolean success = OpenCVLoader.initDebug();
        if (success) {
            Toast.makeText(this.getApplicationContext(), "Loading OpenCV Libraries...", Toast.LENGTH_LONG).show();
        } else {
            Toast.makeText(this.getApplicationContext(), "WARNING: Could not load OpenCV Libraries!", Toast.LENGTH_LONG).show();
        }
    }

    // This function is used to deal with the template picture
    // input: the raw template picture
    // output: Display the processed pictures and mark the reaction area and standard color chart
    // return: Get the color matrix of 24 standard color chart
    public void deal_pic_template(@NonNull Mat mat) {

        Mat new_mat;
        Mat gray = new Mat();
        Mat temp = new Mat();
        Mat threshold_mat = new Mat();
        Mat hist = new Mat();
        Mat hierarchy = new Mat();
        double h, w;
        int threshold = 0;
        List<Mat> listOfMat = new ArrayList<>();
        ArrayList<Integer> difference = new ArrayList<>();
        ArrayList<double[]> center_circle = new ArrayList<>();
        ArrayList<double[]> circles = new ArrayList<>();
        MatOfFloat range = new MatOfFloat(0, 255);
        MatOfInt histSize = new MatOfInt(256);
        MatOfInt channel = new MatOfInt(0);
        ArrayList<MatOfPoint> cnts = new ArrayList<>();

        // Control the direction of image
        if (mat.size().height > mat.size().width) {
            Core.rotate(mat,mat,Core.ROTATE_90_COUNTERCLOCKWISE);
        }

        mat = new Mat(mat, new Rect(520, 210, 1480, 1480));
        new_mat = mat;

        resultBitmap = Bitmap.createBitmap(new_mat.width(), new_mat.height(), Bitmap.Config.ARGB_8888);

        // Preprocess images
        Imgproc.cvtColor(new_mat, new_mat, Imgproc.COLOR_BGR2RGB);
        Imgproc.cvtColor(new_mat, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(gray, temp, 3);
        h = gray.size().height / 400;
        w = gray.size().width / 400;
        Imgproc.resize(temp, temp, new Size(400, 400));
        listOfMat.add(temp);

        Imgproc.calcHist(listOfMat, channel, new Mat(), hist, histSize, range);
        for (int i=0; i<255; i++) {
            double a = hist.get(i + 1,0)[0];
            double b = hist.get(i,0)[0];
            difference.add(Math.abs((int)a - (int)b));
        }
        int i = 0;
        while (i <= 244) {
            int j = 1;
            while (j <= 25) {
                int we = i + j;
                if (difference.get(we) >= 70) {break;}
                else {j++;}
            }
            if (j == 26) {
                threshold = i;
                break;
            }
            else {i += j;}
        }

        //  Find the circle of the reaction area (large circle)
        Imgproc.threshold(temp, threshold_mat, threshold, 255, Imgproc.THRESH_BINARY);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        for (int x=0; x<4; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel);
        }
        for (int x=0; x<2; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_DILATE, kernel);
        }
        Imgproc.findContours(threshold_mat, cnts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        Point center = new Point();
        float [] radius = new float [1];

        for (MatOfPoint cnt: cnts) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            if (Imgproc.contourArea(cnt) > 5000) {

                Imgproc.minEnclosingCircle(cnt2f, center, radius);
                double[] t = {center.x, center.y, (double)radius[0]};
                center_circle.add(t);
            }
        }
        if (center_circle.size() != 1) {
            System.out.println("Cannot found center circle, process ended");
        }

        main_circle = new Mat(mat, new Rect((int)(h*(center.x-radius[0])), (int)(w*(center.y-radius[0])), (int)(0.5*(h + w)*radius[0]*2), (int)(0.5*(h + w)*radius[0]*2)));

        // Find the small circle of the standard color chart
        Imgproc.threshold(temp, threshold_mat, threshold, 255, Imgproc.THRESH_BINARY);
        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        for (int x=0; x<2; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel2);
        }
        for (int x=0; x<1; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_DILATE, kernel2);
        }
        for (int x=0; x<3; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel2);
        }
        Imgproc.findContours(threshold_mat, cnts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        for (MatOfPoint cnt: cnts) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            if (Imgproc.contourArea(cnt) > 300) {
                if (Imgproc.contourArea(cnt) < 1800) {
                    Point y = new Point();
                    float [] radius_s = new float [1];
                    Imgproc.minEnclosingCircle(cnt2f, y, radius_s);
                    if ((y.x > 10 && y.x < 390) && (y.y > 10 && y.y < 390)) {
                        double[] q = {y.x, y.y, (double)radius_s[0]};
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
        double [][] temp_lst = circles.toArray(new double[24][]);

        // Sort the circles found
        double [][] group1 = new double[5][3];
        double [][] group2 = new double[5][3];
        double [][] group3 = new double[4][3];
        double [][] group4 = new double[5][3];
        double [][] group5 = new double[5][3];

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

        // Calculate the RGB mean value in the circular area and fill in the means matrix
        for (int ron = 0; ron < 24; ron++) {
            double num1, num2, num3;
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
            template_matrix[ron][0] = num1;
            template_matrix[ron][1] = num2;
            template_matrix[ron][2] = num3;
        }

        // Mark the big and small circles in the picture
        for (int x = 0; x < circles.size(); x++) {
            Imgproc.circle(new_mat, new Point(h * Math.round(temp_lst[x][0]), w * Math.round(temp_lst[x][1])),
                    ((int)(0.5 * (h + w) * Math.round(temp_lst[x][2]))-30), new Scalar(255, 0, 0), 2);
            Imgproc.putText(new_mat, String.valueOf(x), new Point(h * Math.round(temp_lst[x][0]), w * Math.round(temp_lst[x][1])),
                    4, 2, new Scalar(255,0,0), 2);
        }
        for (int x = 0; x < center_circle.size(); x++) {
            Imgproc.circle(new_mat, new Point((int)(h * Math.round(center_circle.get(x)[0])), (int)(w * Math.round(center_circle.get(x)[1]))),
                    ((int)(0.5 * (h + w) * Math.round(center_circle.get(x)[2]))), new Scalar(0, 255, 0), 2);
        }
        // Display the marked image
        Utils.matToBitmap(new_mat, resultBitmap);
        iv1.setImageBitmap(resultBitmap);
    }

    // This function is used to deal with the example picture
    // input: the raw example picture
    // return: Get the color matrix of 24 standard color chart
    public void deal_pic_example(@NonNull Mat mat) {
        Mat new_mat;
        Mat gray = new Mat();
        Mat temp = new Mat();
        Mat threshold_mat = new Mat();
        Mat hist = new Mat();
        Mat hierarchy = new Mat();
        int threshold = 0;
        List<Mat> listOfMat = new ArrayList<>();
        ArrayList<Integer> difference = new ArrayList<>();
        ArrayList<double[]> circles = new ArrayList<>();
        MatOfFloat range = new MatOfFloat(0, 255);
        MatOfInt histSize = new MatOfInt(256);
        MatOfInt channel = new MatOfInt(0);
        ArrayList<MatOfPoint> cnts = new ArrayList<>();

        if (mat.size().height > mat.size().width) {
            Core.rotate(mat,mat,Core.ROTATE_90_COUNTERCLOCKWISE);
        }

        mat = new Mat(mat, new Rect(520, 210, 1480, 1480));
        new_mat = mat;

        // Preprocess images
        Imgproc.cvtColor(new_mat, new_mat, Imgproc.COLOR_BGR2RGB);
        Imgproc.cvtColor(new_mat, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(gray, temp, 3);

        Imgproc.resize(temp, temp, new Size(400, 400));
        listOfMat.add(temp);

        Imgproc.calcHist(listOfMat, channel, new Mat(), hist, histSize, range);
        for (int i=0; i<255; i++) {
            double a = hist.get(i + 1,0)[0];
            double b = hist.get(i,0)[0];
            difference.add(Math.abs((int)a - (int)b));
        }
        int i = 0;
        while (i <= 244) {
            int j = 1;
            while (j <= 25) {
                int we = i + j;
                if (difference.get(we) >= 70) {break;}
                else {j++;}
            }
            if (j == 26) {
                threshold = i;
                break;
            }
            else {i += j;}
        }

        // Find the small circle of the standard color chart
        Imgproc.threshold(temp, threshold_mat, threshold, 255, Imgproc.THRESH_BINARY);
        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
        for (int x=0; x<2; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel2); // 腐蚀
        }
        for (int x=0; x<1; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_DILATE, kernel2); // 膨胀
        }
        for (int x=0; x<3; x++) {
            Imgproc.morphologyEx(threshold_mat, threshold_mat, Imgproc.MORPH_ERODE, kernel2); // 腐蚀
        }
        Imgproc.findContours(threshold_mat, cnts, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        for (MatOfPoint cnt: cnts) {
            MatOfPoint2f cnt2f = new MatOfPoint2f(cnt.toArray());
            if (Imgproc.contourArea(cnt) > 300) {
                if (Imgproc.contourArea(cnt) < 1800) {
                    Point y = new Point();
                    float [] radius_s = new float [1];
                    Imgproc.minEnclosingCircle(cnt2f, y, radius_s);
                    if ((y.x > 10 && y.x < 390) && (y.y > 10 && y.y < 390)) {
                        double[] q = {y.x, y.y, (double)radius_s[0]};
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
        double [][] temp_lst = circles.toArray(new double[24][]);

        // Sort the circles found
        double [][] group1 = new double[5][3];
        double [][] group2 = new double[5][3];
        double [][] group3 = new double[4][3];
        double [][] group4 = new double[5][3];
        double [][] group5 = new double[5][3];

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


        // Calculate the RGB mean value in the circular area and fill in the means matrix
        for (int ron = 0; ron < 24; ron++) {
            double num1, num2, num3;
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
            example_matrix[ron][0] = num1;
            example_matrix[ron][1] = num2;
            example_matrix[ron][2] = num3;

        }
    }

    // This function is used to obtain the color correction matrix for the reaction area
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

    // Color correct the reaction area and get the final result
    // input: Picture of reaction area and color correction matrix
    // output: Calculated concentration
    public void Reaction_part(@NonNull Mat cir, double[][] x) {
        Mat kernel = new Mat(3,3, CvType.CV_32F);
        Mat hsv_mat = new Mat();
        Mat gray_mat = new Mat();
        Matrix temp_x = new Matrix(x);
        double h = cir.height();
        double w = cir.width();
        double center = h/2;
        double R=0,G=0,S=0,GR=0, number=0;
        double Concentration;

        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                kernel.get(i,j)[0] = x[i][j];
            }
        }
        // Apply the color correction matrix to the image
        for (int i=0; i<h; i++) {
            for (int j=0; j<w; j++) {
                Matrix temp_color = new Matrix(cir.get(i, j), 1);
                temp_color = temp_color.times(temp_x);
                double[][] temp = (temp_color).getArray();
                cir.get(i, j)[0] = temp[0][0];
                cir.get(i, j)[1] = temp[0][1];
                cir.get(i, j)[2] = temp[0][2];
            }
        }

        // Imgproc.filter2D(cir, cir, -1, kernel, new Point(-1,-1), 0, Core.BORDER_DEFAULT);
        // show reaction area
        resultBitmap = Bitmap.createBitmap(cir.width(), cir.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(cir, resultBitmap);
        iv1.setImageBitmap(resultBitmap);

        Imgproc.cvtColor(cir, hsv_mat, Imgproc.COLOR_RGB2HSV);
        Imgproc.cvtColor(cir, gray_mat, Imgproc.COLOR_RGB2GRAY);

        // Extract the value of R G S gray channel
        for (int a=0; a<h; a++) {
            for (int b=0; b<w; b++) {
                if (Math.pow(a-center, 2) + Math.pow(b-center*1.5, 2) < Math.pow(center*0.5, 2)) {
                    R += cir.get(a, b)[0];
                    G += cir.get(a, b)[1];
                    S += hsv_mat.get(a, b)[1];
                    GR += gray_mat.get(a, b)[0];
                    number ++;
                }
            }
        }
        R = R/number;
        G = G/number;
        S = S/number;
        GR = GR/number;

        // Formula to calculate concentration
        Concentration = 3.44*R - 11.98*G + 12.04*GR + 6.32*S - 722.95;

        System.out.println(Concentration);
        String cc = String.format("%.2f",Concentration);
        tv1.setTextSize(30);
        if (Concentration < 0) {
            tv1.setText("Final Concentration: 0");
        }
        else if (Concentration > 100) {
            tv1.setText("Final Concentration: 100");
        }
        else {
            tv1.setText("Final Concentration: " + cc);
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}


