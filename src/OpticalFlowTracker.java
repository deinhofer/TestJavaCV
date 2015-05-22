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
    private static final int MAX_CORNERS = 6;
	private static CanvasFrame frame;
	private static FaceDetection faceDetection=new FaceDetection();
	private static CvRect roiRect=null;
	private static CvRect faceRect=null;
	
    static int win_size = 15;
    static IntPointer corner_count;
    static int flags=0;

    IplImage[] imgGrey=new IplImage[2];
    IplImage[] imgPyr=new IplImage[2];

    

    public static void main(String[] args) throws Exception {
		Loader.load(opencv_objdetect.class);

    	corner_count=new IntPointer(1).put(MAX_CORNERS);
		FrameGrabber grabber = FrameGrabber.create(FrameGrabber.list.get(6),0);
		grabber.setImageWidth(320);
		grabber.setImageHeight(240);
		
		for(int n=0;n<4;n++) {
		frame=new CanvasFrame("Test");
		grabber.start();

		boolean running=true;
		IplImage img=grabber.grab();
		cvFlip(img, img, 1);

		int width  = img.width();
		int height = img.height();
		roiRect= cvRect(0,0,width,height);
		IplImage imgA    = IplImage.create(width, height, IPL_DEPTH_8U, 1);
		cvCvtColor(img, imgA, CV_BGR2GRAY);
		flags=0;
		
        CvSize pyr_sz = cvSize(imgA.width(), imgA.height());

        IplImage pyrA = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
        IplImage pyrB = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
		CvPoint2D32f cornersA=null;
		
		//while(running) {
		for(int f=0;f<200;f++) {
			
			img=grabber.grab();
			cvFlip(img, img, 1);

			IplImage imgB    = IplImage.create(width, height, IPL_DEPTH_8U, 1);
			cvCvtColor(img, imgB, CV_BGR2GRAY);
			cvSetImageROI(imgB,roiRect);

			if(cornersA==null || corner_count.get()==0) {
				cvResetImageROI(imgA);
				cornersA = findFeatures(imgA);
				System.out.println("after findFeatures: corner_count.get(): "
						+ corner_count.get());
			}
			//buildOpticalFlowPyramid(imgA,pyrA,win_size,0,true,0,0,false);
			if(cornersA!=null && corner_count.get()>0) {
				cvSetImageROI(imgA,roiRect);
				cvSetImageROI(imgB,roiRect);
				cvSetImageROI(pyrA,roiRect);
				cvSetImageROI(pyrB,roiRect);
				//cvSetImageROI(img,roiRect);
				
				CvPoint2D32f cornersB = trackOpticalFlow(imgA, imgB, cornersA,
						pyrA, pyrB);
				cornersA = cornersB;
				flags|=CV_LKFLOW_PYR_A_READY;
				
				drawPoints(img,cornersB);
			}
			
	        frame.showImage(img);

			imgA=imgB;			
			//Pointer.memcpy(cornersA, cornersB, cornersA.sizeof());
			pyrA=pyrB;
			//flags|=CV_LKFLOW_PYR_A_READY;
			
		}
		grabber.stop();
		frame.dispose();

		}
		
/*

        cvSaveImage(
                "image0-1.png",
                imgC);
        cvNamedWindow( "LKpyr_OpticalFlow", 0 );
        cvShowImage( "LKpyr_OpticalFlow", imgC );
        cvWaitKey(0);
        */
    }
    
    public static CvPoint2D32f trackOpticalFlow(IplImage imgA, IplImage imgB, CvPoint2D32f cornersA, IplImage pyrA, IplImage pyrB) {
        // Call Lucas Kanade algorithm
        BytePointer features_found = new BytePointer(MAX_CORNERS);
        FloatPointer feature_errors = new FloatPointer(MAX_CORNERS);

        CvPoint2D32f cornersB = new CvPoint2D32f(MAX_CORNERS);
        cvCalcOpticalFlowPyrLK(imgA, imgB, pyrA, pyrB, cornersA, cornersB,
                corner_count.get(), cvSize(win_size, win_size), 5,
                features_found, feature_errors,
                cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3),flags);
        
        return rejectBadPoints(cornersA, cornersB, features_found, feature_errors);
    }
    
    private static CvPoint2D32f rejectBadPoints(CvPoint2D32f cornersA, CvPoint2D32f cornersB, BytePointer features_found, FloatPointer feature_errors) {
        int newCornerCount=0;
        CvPoint2D32f newCorners = new CvPoint2D32f(MAX_CORNERS);
                
        // Make an image of the results
        for (int i = 0; i < corner_count.get(); i++) {
            if (features_found.get(i) == 0 || feature_errors.get(i) > 550) {
                System.out.println("Error is " + feature_errors.get(i) + "/n");
                continue;
            }
            //System.out.println("Got it/n");
            cornersA.position(i);
            cornersB.position(i);
            
            newCorners.position(newCornerCount);
            newCorners.put((double)cornersB.x(),(double)cornersB.y());
            newCornerCount++;
            
        }    	

        corner_count.put(newCornerCount);
        newCorners.position(0);
        System.out.println("new corner_count: "+corner_count.get());

        return newCorners;
    }
    
    public static void drawPoints(IplImage img, CvPoint2D32f corners) {   	
        for (int i = 0; i < corner_count.get(); i++) {
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

				CvPoint2D32f cornersA = new CvPoint2D32f(MAX_CORNERS);
				CvArr mask = null;
				
				cornersA=addCvPoints(cornersA,initNose);
				cornersA=addCvPoints(cornersA,initChin);
				corner_count.put(cornersA.position());
				cornersA.position(0);
				
				//Setting image ROI should improve tracking quality, but disabled it so far
				cvSetImageROI(imgA,roiRect);
				//cvSetImageROI(imgA,faceRect);
				//cvSetImageROI(imgA,faceRect);
				
				//Uses given points and tries to find better trackable ones in the neighbourhood.
				//cvTermCriteria is set to 1, 1 because otherwise the new points would be too far away. 
				cvFindCornerSubPix(
						imgA,
						cornersA,
						corner_count.get(),
						cvSize(win_size, win_size),
						cvSize(-1, -1),
//						cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
				cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1, 1));
						
				//cvResetImageROI(imgA);

				System.out.println("Found trackable points: cornersA: "+cornersA);
				return cornersA;
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
