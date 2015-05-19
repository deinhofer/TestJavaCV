/*
 * Because I believe that examples are the easiest way how to use JavaCV, I am 
 * sending a sample based on http://dasl.mem.drexel.edu/~noahKuntz/openCVTut9.html
 *
 * burgetrm@gmail.com
 */

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.FrameGrabber.Exception;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class OpticalFlowTracker {
    private static final int MAX_CORNERS = 50;
	private static CanvasFrame frame=new CanvasFrame("Test");
	
    static int win_size = 15;
    static IntPointer corner_count;
    static int flags=0;



    public static void main(String[] args) throws Exception {
        // Load two images and allocate other structures
/*        IplImage imgA = cvLoadImage(
                "image0.png",
                CV_LOAD_IMAGE_GRAYSCALE);
        IplImage imgB = cvLoadImage(
                "image1.png",
                CV_LOAD_IMAGE_GRAYSCALE);
*/
		Loader.load(opencv_objdetect.class);

    	corner_count=new IntPointer(1).put(MAX_CORNERS);
		FrameGrabber grabber = FrameGrabber.create(FrameGrabber.list.get(6),0);
		grabber.start();

		boolean running=true;
		IplImage img=grabber.grab();

		int width  = img.width();
		int height = img.height();
		IplImage imgA    = IplImage.create(width, height, IPL_DEPTH_8U, 1);
		cvCvtColor(img, imgA, CV_BGR2GRAY);
		flags=0;
		
        CvSize pyr_sz = cvSize(imgA.width() + 8, imgA.height() / 3);

        IplImage pyrA = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
        IplImage pyrB = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
		CvPoint2D32f cornersA=findFeatures(imgA);

		while(running) {
			
			img=grabber.grab();
			IplImage imgB    = IplImage.create(width, height, IPL_DEPTH_8U, 1);
			cvCvtColor(img, imgB, CV_BGR2GRAY);

			if(corner_count.get()==0) {
				cornersA = findFeatures(imgA);
				System.out.println("after findFeatures: corner_count.get(): "
						+ corner_count.get());
			}
			//buildOpticalFlowPyramid(imgA,pyrA,win_size,0,true,0,0,false);
			CvPoint2D32f cornersB=imageGrabbed(imgA, imgB, img,cornersA,pyrA, pyrB);
			imgA=imgB;
			cornersA=cornersB;
			//Pointer.memcpy(cornersA, cornersB, cornersA.sizeof());
			pyrA=pyrB;
			//flags|=CV_LKFLOW_PYR_A_READY;
		}
		grabber.stop();
		frame.dispose();
/*

        cvSaveImage(
                "image0-1.png",
                imgC);
        cvNamedWindow( "LKpyr_OpticalFlow", 0 );
        cvShowImage( "LKpyr_OpticalFlow", imgC );
        cvWaitKey(0);
        */
    }
    
    public static CvPoint2D32f imageGrabbed(IplImage imgA, IplImage imgB, IplImage img, CvPoint2D32f cornersA, IplImage pyrA, IplImage pyrB) {

        // IplImage imgC = cvLoadImage("OpticalFlow1.png",
        // CV_LOAD_IMAGE_UNCHANGED);
        /*
        IplImage imgC = cvLoadImage(
                "image0.png",
                CV_LOAD_IMAGE_UNCHANGED);
        */
        
        

        // Call Lucas Kanade algorithm
        BytePointer features_found = new BytePointer(MAX_CORNERS);
        FloatPointer feature_errors = new FloatPointer(MAX_CORNERS);

        CvPoint2D32f cornersB = new CvPoint2D32f(MAX_CORNERS);
        cvCalcOpticalFlowPyrLK(imgA, imgB, pyrA, pyrB, cornersA, cornersB,
                corner_count.get(), cvSize(win_size, win_size), 5,
                features_found, feature_errors,
                cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),flags);

        System.out.println("corner_count.get(): "+corner_count.get());
        // Make an image of the results
        for (int i = 0; i < corner_count.get(); i++) {
            if (features_found.get(i) == 0 || feature_errors.get(i) > 550) {
                System.out.println("Error is " + feature_errors.get(i) + "/n");
                continue;
            }
            System.out.println("Got it/n");
            cornersA.position(i);
            cornersB.position(i);
            CvPoint p0 = cvPoint(Math.round(cornersA.x()),
                    Math.round(cornersA.y()));
            CvPoint p1 = cvPoint(Math.round(cornersB.x()),
                    Math.round(cornersB.y()));
            
            cvLine(img, p0, p1, CV_RGB(255, 0, 0), 
                    2, 8, 0);
        }
        frame.showImage(img);
        return cornersB;
    }
    
    public static CvPoint2D32f findFeatures(IplImage imgA) {
        CvSize img_sz = cvGetSize(imgA);
        
        // Get the features for tracking
        IplImage eig_image = IplImage.create(img_sz, IPL_DEPTH_8U, 1);
        IplImage tmp_image = IplImage.create(img_sz, IPL_DEPTH_8U, 1);

        CvPoint2D32f cornersA = new CvPoint2D32f(MAX_CORNERS);

        CvArr mask = null;
        cvGoodFeaturesToTrack(imgA, eig_image, tmp_image, cornersA,
                corner_count, 0.05, 5.0, mask, 3, 0, 0.04);

        cvFindCornerSubPix(imgA, cornersA, corner_count.get(),
                cvSize(win_size, win_size), cvSize(-1, -1),
                cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
        
        return cornersA;
    }
}
