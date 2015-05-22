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
import org.bytedeco.javacpp.opencv_core.CvPoint;
import org.bytedeco.javacpp.opencv_core.CvPoint2D32f;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.*;
import org.bytedeco.javacv.FrameGrabber.Exception;

import static org.bytedeco.javacpp.helper.opencv_core.CV_RGB;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_highgui.*;

public class OpticalFlowTracker {
    private static final int MAX_POINTS = 6;
	private static CanvasFrame frame;
	private static FaceDetection faceDetection=new FaceDetection();
	private static CvRect roiRect=null;
	private static CvRect faceRect=null;
	
    static int winSize = 15;
    static IntPointer nrPoints;
    static int flags=0;
    private static final int A=0;
    private static final int B=1;


    public static void main(String[] args) throws Exception {
		Loader.load(opencv_objdetect.class);

		//Init
	    IplImage[] imgGrey=new IplImage[2];
	    IplImage[] imgPyr=new IplImage[2];
	    CvPoint2D32f[] points=new CvPoint2D32f[2];
	    
    	nrPoints=new IntPointer(1).put(MAX_POINTS);
		FrameGrabber grabber = FrameGrabber.create(FrameGrabber.list.get(6),0);
		grabber.setImageWidth(320);
		grabber.setImageHeight(240);
		
		frame=new CanvasFrame("Test");
		grabber.start();

		boolean running=true;
		IplImage img=grabber.grab();
		cvFlip(img, img, 1);

		int width  = img.width();
		int height = img.height();
		roiRect= cvRect(0,0,width,height);
		imgGrey[A]    = IplImage.create(width, height, IPL_DEPTH_8U, 1);
		cvCvtColor(img, imgGrey[A], CV_BGR2GRAY);
		flags=0;
		
        CvSize pyr_sz = cvSize(imgGrey[A].width(), imgGrey[A].height());

        imgPyr[A] = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
        imgPyr[B] = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
		points[A]=null;
		
		while(running) {			
			img=grabber.grab();
			cvFlip(img, img, 1);

			imgGrey[B]    = IplImage.create(width, height, IPL_DEPTH_8U, 1);
			cvCvtColor(img, imgGrey[B], CV_BGR2GRAY);
			cvSetImageROI(imgGrey[B],roiRect);

			if(points[A]==null || nrPoints.get()==0) {
				cvResetImageROI(imgGrey[A]);
				points[A] = findFeatures(imgGrey[A]);
				System.out.println("after findFeatures: nrPoints.get(): "
						+ nrPoints.get());
			}
			if(points[A]!=null && nrPoints.get()>0) {
				cvSetImageROI(imgGrey[A],roiRect);
				cvSetImageROI(imgGrey[B],roiRect);
				cvSetImageROI(imgPyr[A],roiRect);
				cvSetImageROI(imgPyr[B],roiRect);
				//cvSetImageROI(img,roiRect);
				
				points[B] = trackOpticalFlow(imgGrey, imgPyr, points[A]);
				points[A] = points[B];
				flags|=CV_LKFLOW_PYR_A_READY;
				
				drawPoints(img,points[B]);
			}
			
	        frame.showImage(img);

			imgGrey[A]=imgGrey[B];			
			imgPyr[A]=imgPyr[B];
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
    
    public static CvPoint2D32f trackOpticalFlow(IplImage[] imgGrey, IplImage[] imgPyr, CvPoint2D32f pointsA) {
        // Call Lucas Kanade algorithm
        BytePointer features_found = new BytePointer(MAX_POINTS);
        FloatPointer feature_errors = new FloatPointer(MAX_POINTS);

        CvPoint2D32f pointsB = new CvPoint2D32f(MAX_POINTS);
        cvCalcOpticalFlowPyrLK(imgGrey[A], imgGrey[B], imgPyr[A], imgPyr[B], pointsA, pointsB,
                nrPoints.get(), cvSize(winSize, winSize), 5,
                features_found, feature_errors,
                cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),flags);
        
        return rejectBadPoints(pointsA, pointsB, features_found, feature_errors);
    }
    
    private static CvPoint2D32f rejectBadPoints(CvPoint2D32f pointsA, CvPoint2D32f pointsB, BytePointer features_found, FloatPointer feature_errors) {
        int newCornerCount=0;
        CvPoint2D32f newCorners = new CvPoint2D32f(MAX_POINTS);
                
        // Make an image of the results
        for (int i = 0; i < nrPoints.get(); i++) {
            if (features_found.get(i) == 0 || feature_errors.get(i) > 550) {
                System.out.println("Error is " + feature_errors.get(i) + "/n");
                continue;
            }
            //System.out.println("Got it/n");
            pointsA.position(i);
            pointsB.position(i);
            
            newCorners.position(newCornerCount);
            newCorners.put((double)pointsB.x(),(double)pointsB.y());
            newCornerCount++;
            
        }    	

        nrPoints.put(newCornerCount);
        pointsA.position(0);
        pointsB.position(0);
        newCorners.position(0);
        System.out.println("new nrPoints: "+nrPoints.get());

        return newCorners;
    }
    
    public static void drawPoints(IplImage img, CvPoint2D32f corners) {   	
        for (int i = 0; i < nrPoints.get(); i++) {
        	corners.position(i);
			CvPoint p = cvPoint(Math.round(corners.x()),
					Math.round(corners.y()));
			System.out.println("p"+i+": " + p);
			// cvLine(img, p0, p1, CV_RGB(255, 0, 0),
			// 2, 8, 0);
			cvCircle(img, p, 4, CV_RGB(255, 0, 0), 2, 5, 0);
        }
        corners.position(0);
    }
    
    public static CvPoint2D32f findFeatures(IplImage imgA) {
		
		try {
			faceRect = faceDetection.detectFace(imgA);
			if(faceRect!=null) { 
				System.out.println("Found face at: "+faceRect);
				
				//roiRect=faceRect;
				
				int x = faceRect.x() + faceRect.width() / 2;
				int y = faceRect.y() + faceRect.height() / 2;
				CvPoint initNose[]=new CvPoint[]{cvPoint(x,y-10),cvPoint(x-20,y-10),cvPoint(x+20,y-10),cvPoint(x,y+10)};
				CvPoint initChin[]=new CvPoint[]{cvPoint(x,y+65),cvPoint(x,y+55)};

				CvPoint2D32f pointsA = new CvPoint2D32f(MAX_POINTS);
				CvArr mask = null;
				
				pointsA=addCvPoints(pointsA,initNose);
				pointsA=addCvPoints(pointsA,initChin);
				nrPoints.put(pointsA.position());
				pointsA.position(0);
				
				//Setting image ROI should improve tracking quality, but disabled it so far
				cvSetImageROI(imgA,roiRect);
				//cvSetImageROI(imgA,faceRect);
				//cvSetImageROI(imgA,faceRect);
				
				//Uses given points and tries to find better trackable ones in the neighbourhood.
				//cvTermCriteria is set to 1, 1 because otherwise the new points would be too far away. 
				cvFindCornerSubPix(
						imgA,
						pointsA,
						nrPoints.get(),
						cvSize(winSize, winSize),
						cvSize(-1, -1),
//						cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
				cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1, 1));
						
				//cvResetImageROI(imgA);

				System.out.println("Found trackable points: pointsA: "+pointsA);
				return pointsA;
			}			
		} catch (java.lang.Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
    }
    
    private static CvPoint2D32f addCvPoints(CvPoint2D32f corners, CvPoint[] points) {
    	for(CvPoint point : points) {
    		corners.put(point);
    		corners.position(corners.position()+1);
    	}
    	return corners;
    }
}
