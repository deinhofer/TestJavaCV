import java.io.File;
import java.net.URL;

import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.opencv_core.CvPoint2D32f;

import static org.bytedeco.javacpp.ARToolKitPlus.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;

public class Demo {
    public static void main(String[] args) throws Exception {
    	
    	int frameGrabberIdx=4;
    	if(args!=null && args.length > 0 && args[0]!=null && !"".equals(args[0])) {
    		frameGrabberIdx=Integer.parseInt(args[0]);
    	}

    	int camIdx=0;
    	if(args!=null && args.length > 0 && args[1]!=null && !"".equals(args[1])) {
    		camIdx=Integer.parseInt(args[1]);
    	}
    	int userWidth=640;
    	if(args!=null && args.length > 0 && args[2]!=null && !"".equals(args[2])) {
    		userWidth=Integer.parseInt(args[2]);
    	}

    	int userHeight=480;
    	if(args!=null && args.length > 0 && args[3]!=null && !"".equals(args[3])) {
    		userHeight=Integer.parseInt(args[3]);
    	}

        String classifierName = null;
        classifierName="haarcascade_frontalface_alt.xml";
        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);

        // We can "cast" Pointer objects by instantiating a new object of the desired class.
        CvHaarClassifierCascade classifier = new CvHaarClassifierCascade(cvLoad(classifierName));
        if (classifier.isNull()) {
            System.err.println("Error loading classifier file \"" + classifierName + "\".");
            System.exit(1);
        }              

        //Test CvPoint32
        CvPoint2D32f p=new CvPoint2D32f(2);
        p.put(2.1,1.5);
        p.position(1).put(3.2,1.6);
        System.out.println("p: "+p);
        
        CvPoint2D32f p2=new CvPoint2D32f(2);
        p.position(0);
        p2.put(p.x(),p.y());
        p2.position(1);
        p.position(1);
        p2.put(p.x(),p.y());
        p.put(4.4,4.5);
        System.out.println("p: "+p);
        System.out.println("p2: "+p2);
        
        // The available FrameGrabber classes include OpenCVFrameGrabber (opencv_highgui),
        // DC1394FrameGrabber, FlyCaptureFrameGrabber, OpenKinectFrameGrabber,
        // PS3EyeFrameGrabber, VideoInputFrameGrabber, and FFmpegFrameGrabber.
        
        System.out.println("List of grabbers (indices): "+FrameGrabber.list);
        System.out.println("Using grabber: "+FrameGrabber.list.get(frameGrabberIdx)+", and camIdx: "+camIdx);
        FrameGrabber grabber = FrameGrabber.create(FrameGrabber.list.get(frameGrabberIdx),camIdx);
        //FFmpegFrameGrabber grabber =new FFmpegFrameGrabber("video=Integrated Camera");
        
        /*
        FFmpegFrameGrabber grabber = new FFmpegFrameGrabber("desktop");
        grabber.setBitsPerPixel(8);
        grabber.setFormat("gdigrab");
        
        grabber.setFrameRate(10);        //FFmpegFrameGrabber grabber =new FFmpegFrameGrabber("0");
        */
        //grabber.setFormat("vfwcap");        
        
        //FrameGrabber grabber = FrameGrabber.createDefault(frameGrabberIdx);
        grabber.setImageHeight(userHeight);
        grabber.setImageWidth(userWidth);
        //grabber.setFrameRate(10);
        grabber.start();
        
        // FAQ about IplImage:
        // - For custom raw processing of data, getByteBuffer() returns an NIO direct
        //   buffer wrapped around the memory pointed by imageData, and under Android we can
        //   also use that Buffer with Bitmap.copyPixelsFromBuffer() and copyPixelsToBuffer().
        // - To get a BufferedImage from an IplImage, we may call getBufferedImage().
        // - The createFrom() factory method can construct an IplImage from a BufferedImage.
        // - There are also a few copy*() methods for BufferedImage<->IplImage data transfers.
        IplImage grabbedImage = grabber.grab();
        int width  = grabbedImage.width();
        int height = grabbedImage.height();
        IplImage grayImage    = IplImage.create(width, height, IPL_DEPTH_8U, 1);

        // Objects allocated with a create*() or clone() factory method are automatically released
        // by the garbage collector, but may still be explicitly released by calling release().
        // You shall NOT call cvReleaseImage(), cvReleaseMemStorage(), etc. on objects allocated this way.
        CvMemStorage storage = CvMemStorage.create();

        // CanvasFrame is a JFrame containing a Canvas component, which is hardware accelerated.
        // It can also switch into full-screen mode when called with a screenNumber.
        // We should also specify the relative monitor/camera response for proper gamma correction.
        CanvasFrame frame = new CanvasFrame("Some Title", CanvasFrame.getDefaultGamma()/grabber.getGamma());

        // We can allocate native arrays using constructors taking an integer as argument.
        CvPoint hatPoints = new CvPoint(3);

        CvRect faceRect=null;
        while (frame.isVisible() && (grabbedImage = grabber.grab()) != null) {
        	
        	//if(faceRect==null) {
	            cvClearMemStorage(storage);
	
	            // Let's try to detect some faces! but we need a grayscale image...
	            cvCvtColor(grabbedImage, grayImage, CV_BGR2GRAY);
/*
	            CvSeq faces = cvHaarDetectObjects(grayImage, classifier, storage,
	                    1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
	                    */
	            
	            CvSeq faces = cvHaarDetectObjects(grayImage, classifier, storage,
	                    1.1, 1, CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT);

	                    
	            int total = faces.total();
	            
	            for (int i = 0; i < total; i++) {
	            	//System.out.println("Face "+i+"detected");
	            	
	                faceRect = new CvRect(cvGetSeqElem(faces, i));
	                int x = faceRect.x(), y = faceRect.y(), w = faceRect.width(), h = faceRect.height();
	                cvRectangle(grabbedImage, cvPoint(x, y), cvPoint(x+w, y+h), CvScalar.RED, 1, CV_AA, 0);
	
	                /*
	                // To access or pass as argument the elements of a native array, call position() before.
	                hatPoints.position(0).x(x-w/10)   .y(y-h/10);
	                hatPoints.position(1).x(x+w*11/10).y(y-h/10);
	                hatPoints.position(2).x(x+w/2)    .y(y-h/2);
	                cvFillConvexPoly(grabbedImage, hatPoints.position(0), 3, CvScalar.GREEN, CV_AA, 0);
	                */
	            }
        	//} else {
        		
        		
        	//}
/*
            // Let's find some contours! but first some thresholding...
            cvThreshold(grayImage, grayImage, 64, 255, CV_THRESH_BINARY);

            // To check if an output argument is null we may call either isNull() or equals(null).
            CvSeq contour = new CvSeq(null);
            cvFindContours(grayImage, storage, contour, Loader.sizeof(CvContour.class),
                    CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
            while (contour != null && !contour.isNull()) {
                if (contour.elem_size() > 0) {
                    CvSeq points = cvApproxPoly(contour, Loader.sizeof(CvContour.class),
                            storage, CV_POLY_APPROX_DP, cvContourPerimeter(contour)*0.02, 0);
                    cvDrawContours(grabbedImage, points, CvScalar.BLUE, CvScalar.BLUE, -1, 1, CV_AA);
                }
                contour = contour.h_next();
            }
  */       
            frame.showImage(grabbedImage);
        }
        frame.dispose();
        //recorder.stop();
        grabber.stop();
    }
}