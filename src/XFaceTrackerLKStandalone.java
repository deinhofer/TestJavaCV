import static org.bytedeco.javacpp.helper.opencv_core.CV_RGB;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

import java.util.Arrays;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_core.CvPoint;
import org.bytedeco.javacpp.opencv_core.CvPoint2D32f;
import org.bytedeco.javacpp.opencv_core.CvRect;
import org.bytedeco.javacpp.opencv_core.CvSize;
import org.bytedeco.javacpp.opencv_core.IplImage;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.FrameGrabber.Exception;



public class XFaceTrackerLKStandalone {
	private static final int GAIN = 25;
	FrameGrabber grabber;
	CanvasFrame frame;
	CvRect faceRect=null;
	FaceDetection faceDetection=new FaceDetection();

	private static final int MAX_CORNERS = 2;

	private IplImage[] imgBGR=new IplImage[MAX_CORNERS];
	private IplImage[] imgGrey=new IplImage[MAX_CORNERS];
	private IplImage[] imgPyr=new IplImage[MAX_CORNERS];
	
	//params LK algorithm
	//nr. of points to track with optical flow algorithm
    int win_size = 11;
    IntPointer corner_count = new IntPointer(1).put(MAX_CORNERS);

	private static final int LAST = 0;
	private static final int NOW = 1;

    CvPoint2D32f[] points = new CvPoint2D32f[MAX_CORNERS];
    
	
    //params for mouse coordinates
	private int lastX=0;
	private int lastY=0;
	
	static boolean running=false;

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		XFaceTrackerLKStandalone tracker=new XFaceTrackerLKStandalone();
		FrameGrabber grabber = FrameGrabber.create(FrameGrabber.list.get(6),0);
		tracker.grabber=grabber;
		tracker.frame=new CanvasFrame("Test");
		tracker.grabber.start();

		running=true;
		while(running) {
			
			IplImage image=tracker.grabber.grab();
			tracker.imageGrabbed(image);
			//image.release();
			//image=null;
			
		}
		tracker.grabber.stop();
		tracker.frame.dispose();
	}

	public XFaceTrackerLKStandalone() {
	}

    /**
     * Callback called when a new frame was grabbed. 
     */
	
	public void imageGrabbed(IplImage image) {
		//System.out.println(".");
				
		try {
			//if not initialized, find face first
			if(faceRect==null) {
				//This will set initial point locations and init last and current image references
				detectFaceAndInit(image);
			} else {

				//convert img to greyscale
				imgBGR[NOW]=image;
				imgGrey[NOW]=faceDetection.convertToGrayScaleIplImage(image);
				//Now calc optical flow and get new locations back.
				
				CvPoint[] pointsNow=calcFlowOfPoints();
				//from drawings on an image
				System.out.println("Drawing new points: "+Arrays.toString(pointsNow));
				
				if (pointsNow != null) {
					for (CvPoint p : pointsNow) {
						cvCircle(image, p, 4, CV_RGB(255, 255, 255), 2, 8, 0);
					}
				}
				//and a very platform dependant Canvas/Frame!!
				faceDetection.drawFaceRect(faceRect, image);

				frame.showImage(image);

				
				//send coordinates to output ports
				if(faceRect!=null) {
					int x = faceRect.x()+faceRect.width()/2;
					int y = faceRect.y()+faceRect.height()/2;
					int relX=(x-lastX)*GAIN*-1;
					int relY=(y-lastY)*GAIN;
					System.out.println("["+relX+", "+relX+"]");
					//opNoseX.sendData(ConversionUtils.intToBytes(relX));
					//opNoseY.sendData(ConversionUtils.intToBytes(relY));

					lastX=x;
					lastY=y;				
				}
				
				imgBGR[LAST]=imgBGR[NOW].clone();
				imgGrey[LAST]=imgGrey[NOW].clone();

			}
		} catch (java.lang.Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private CvPoint[] calcFlowOfPoints() {
		System.out.println("Calculating flow of points...");
        //CvArr mask = null;
        /*
        cvFindCornerSubPix(imgA, points[], corner_count.get(),
                cvSize(win_size, win_size), cvSize(-1, -1),
                cvTermCriteria(CV_TERMCRIT_ITER,1,1.0));
                //cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
*/
        // Call Lucas Kanade algorithm
        BytePointer features_found = new BytePointer(MAX_CORNERS);
        FloatPointer feature_errors = new FloatPointer(MAX_CORNERS);

        cvCalcOpticalFlowPyrLK(imgGrey[LAST],imgGrey[NOW], imgPyr[LAST], imgPyr[NOW], points[LAST], points[NOW],
                corner_count.get(), cvSize(win_size, win_size), 5,
                features_found, feature_errors,
                cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3), 0);

        CvPoint[] pointsNow = {new CvPoint(),new CvPoint()};
        // Make an image of the results
        for (int i = 0; i < MAX_CORNERS; i++) {
            if (features_found.get(i) == 0 || feature_errors.get(i) > 550) {
                System.out.println("Error is " + feature_errors.get(i) + "/n");
                continue;
            }
            
            points[NOW].position(i);
            
            CvPoint p = cvPoint(Math.round(points[NOW].position(i).x()),
                    Math.round(points[NOW].position(i).y()));
            System.out.println("Point["+i+"]: "+p);
            pointsNow[i]=p;
        }
        return pointsNow;
	}
	
	private void detectFaceAndInit(IplImage image) {
		//Strictly seperate, cv algorithms (detection,...)
		try {
			System.out.println("Detecting Face...");
			faceRect=faceDetection.detectFace(image);
			
			if(faceRect!=null) { 
				System.out.println("Found face at: "+faceRect);
				int x = faceRect.x() + faceRect.width() / 2;
				int y = faceRect.y() + faceRect.height() / 2;
				CvPoint initNose=cvPoint(x,y);
				CvPoint initChin=cvPoint(x,y+10);
				
				points[LAST] = new CvPoint2D32f(MAX_CORNERS);
				points[NOW] = new CvPoint2D32f(MAX_CORNERS);

				points[LAST].position(0).put(initNose);
				points[NOW].position(0).put(initNose);
				points[LAST].position(1).put(initChin);
				points[NOW].position(1).put(initChin);

				System.out.println("Points[LAST] before findCornerSubPix: "+points[LAST]);
				System.out.println("Points[NOW] before findCornerSubPix: "+points[NOW]);

				
				imgBGR[NOW] = image;
				imgBGR[LAST]=imgBGR[NOW].clone();
				imgGrey[NOW] = faceDetection
						.convertToGrayScaleIplImage(image);
				imgGrey[LAST]=imgGrey[NOW].clone();
				
		        CvSize pyr_sz = cvSize(imgBGR[LAST].width() + 8, imgBGR[NOW].height() / 3);

		        imgPyr[LAST] = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);
		        imgPyr[NOW] = IplImage.create(pyr_sz, IPL_DEPTH_8U, 1);

     			cvFindCornerSubPix( imgGrey[NOW], points[NOW], corner_count.get(),
						cvSize(win_size,win_size), cvSize(-1,-1),
						//cvTermCriteria(CV_TERMCRIT_ITER,1,1.0));
						cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));
			  
				//also set last points to newly refined points
		
				//points[NOW].memcpy(points[NOW], points[LAST], 2);
				//points[LAST].points[NOW].sizeof();
				points[LAST].position(0);		
				points[NOW].position(0);
				points[LAST].put(points[NOW].x(),points[NOW].y());
				points[LAST].position(1);		
				points[NOW].position(1);
				points[LAST].put(points[NOW].x(),points[NOW].y());
				
				System.out.println("Points[LAST] after findCornerSubPix: "+points[LAST]);
				System.out.println("Points[NOW] after findCornerSubPix: "+points[NOW]);
			}
		} catch (java.lang.Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	

}
