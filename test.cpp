#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>   

#define ledNum 8

using namespace cv;
using namespace std;

int BinarizeImageByOTSU (IplImage * src)  
{   
     
    //get the ROI   
    CvRect rect = cvGetImageROI(src);  
     
    //information of the source image   
    int x = rect.x;  
    int y = rect.y;  
    int width = rect.width;   
    int height = rect.height;  
    int ws = src->widthStep;  
     
    int thresholdValue=1;//��ֵ   
    int ihist [256] ; // ͼ��ֱ��ͼ, 256����   
    int i, j, k,n, n1, n2, Color=0;  
    double m1, m2, sum, csum, fmax, sb;  
    memset (ihist, 0, sizeof (ihist)) ; // ��ֱ��ͼ�� ��...   
     
    for (i=y;i< y+height;i++) // ����ֱ��ͼ   
    {   
        int mul =  i*ws;  
        for (j=x;j<x+width;j++)  
        {   
            //Color=Point (i,j) ;   
            Color = (int)(unsigned char)*(src->imageData + mul+ j);  
            ihist [Color] +=1;  
        }  
    }  
    sum=csum=0.0;  
    n=0;  
    for (k = 0; k <= 255; k++)  
    {   
        sum+= (double) k* (double) ihist [k] ; // x*f (x) ������   
        n +=ihist [k]; //f (x) ����   
    }  
    // do the otsu global thresholding method   
    fmax = - 1.0;  
    n1 = 0;  
    for (k=0;k<255;k++)   
    {  
        n1+=ihist [k] ;  
        if (! n1)  
        {   
            continue;   
        }  
        n2=n- n1;  
        if (n2==0)   
        {  
            break;  
        }  
        csum+= (double) k*ihist [k] ;  
        m1=csum/ n1;  
        m2= (sum- csum) /n2;  
        sb = ( double) n1* ( double) n2* ( m1 - m2) * (m1- m2) ;  
         
        if (sb>fmax)   
        {  
            fmax=sb;  
            thresholdValue=k;  
        }  
    }  
     
    //binarize the image    
    cvThreshold( src, src ,thresholdValue, 255, CV_THRESH_BINARY );   
    return 0;  
}   

bool rectA_intersect_rectB(CvRect rectA, Rect rectB)
{
	//if ( rectA.x > rectB.x + rectB.width ) { return false; }  
	//if ( rectA.y > rectB.y + rectB.height ) { return false; }
	//if ( (rectA.x + rectA.width) < rectB.x ) { return false; }
	//if ( (rectA.y + rectA.height) < rectB.y ) { return false; }
	//float colInt =  min(rectA.x+rectA.width,rectB.x+rectB.width) - max(rectA.x, rectB.x);  
	//float rowInt =  min(rectA.y+rectA.height,rectB.y+rectB.height) - max(rectA.y,rectB.y);  
	//float intersection = colInt * rowInt;  
	//float areaA = rectA.width * rectA.height;  
	//float areaB = rectB.width * rectB.height;  
	//float intersectionPercent =  intersection / (areaA + areaB - intersection);  
	//if ( (0 < intersectionPercent)&&(intersectionPercent < 1)&&(intersection != areaA)&&(intersection != areaB) )
	//{
	//	return true;
	//}
	if(rectA.x >= rectB.x && (rectA.x + rectA.width) <= (rectB.x+ rectB.width) &&
		rectA.y >= rectB.y && (rectA.y + rectA.height) <= (rectB.y+ rectB.height))
	{
		return true;
	}
	else
		return false;
}

void calRect(CvRect rect, CvRect *outRect)
{	 
	int x = outRect->x, y = outRect->y;
	if(rect.x < outRect->x)
	{
		outRect->x = rect.x;
		outRect->width += (x - rect.x);
	}
	if(rect.y < outRect->y)
	{
		outRect->y = rect.y;
		outRect->height += (y - rect.y);
	}
	if(rect.x + rect.width > outRect->x + outRect->width)
		outRect->width = rect.x + rect.width - outRect->x;
	if(rect.y + rect.height  > outRect->y + outRect->height)
		outRect->height = rect.y + rect.height - outRect->y;
}

class LEDCircle
{
private:
	CvPoint2D32f center;  
	float radius;  
	CvScalar scalar;
	bool ifRedLED;
public:
	void calCircle(CvSeq* contour, CvRect outRect)
	{
		cvMinEnclosingCircle(contour,&center,&radius); 
		center.x += outRect.x;
		center.y += outRect.y;
	}

	void calScalar(IplImage* img)
	{
		scalar = cvGet2D(img, center.y, center.x);
		if(scalar.val[2] >= 150)
		{
			ifRedLED = true;
		}
		else
			ifRedLED = false;
	}

	void drawCircle(IplImage* img)
	{
		if(ifRedLED)
			cvCircle(img, cvPointFrom32f(center), cvRound(radius), CV_RGB(255, 0, 0)); 
		else
			cvCircle(img, cvPointFrom32f(center), cvRound(radius), CV_RGB(0, 255, 0)); 
	}

	inline bool operator < (const LEDCircle &right) const
	{
		//cout << this->center.x << "|||" << this->center.y << endl;
		//cout << right.center.x << "|||" << right.center.y << endl;
		//cout << this->radius << endl;
		//cout << "======================="<<endl;

		if(abs(this->center.y - right.center.y) < (this->radius + right.radius) / 2)
		{
			if(this->center.x < right.center.x)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			if(this->center.y < right.center.y)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
	}
};

vector<CvSeq* > calContour(IplImage* gray, const double accuracy)
{
	//�Ҷ�ͼ-��ֵ��-������-ͳ�����-�������ɸѡ
	BinarizeImageByOTSU(gray);
	CvSeq* contourSeq = 0;
	CvMemStorage* storage = cvCreateMemStorage(0);  
    cvFindContours(gray, storage, &contourSeq, sizeof(CvContour), CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE );  

	multimap<double, CvSeq*> area;
    for(; contourSeq != 0; contourSeq = contourSeq->h_next)  
    {
		CvRect rect = cvBoundingRect(contourSeq, 0);
		area.insert(make_pair(cvContourArea(contourSeq), contourSeq));	
    } 
	multimap<double, CvSeq*>::iterator iter;
	iter = area.begin();  
	for(int i = 0; i < area.size() / 2; ++i)
		++iter;
	double midArea = (*iter).first;
	CvRect outRect = cvRect(gray->width, gray->height, 0, 0);

	vector<CvSeq* > contourVector;
	for(iter = area.begin(); iter != area.end(); ++iter)
	{
		if((*iter).first >= (midArea * (1 - accuracy)) && (*iter).first <= (midArea * (1 + accuracy)))
		{
			contourVector.push_back((*iter).second);
		}
	}	
	return contourVector;
}

int main( int argc, char* argv[])  
{  
    IplImage* img0; 
    IplImage* img;
    IplImage* src;  
    if((img0=cvLoadImage("./3.jpg", 1)))//����ͼ��   
    {  
 
        img = cvCreateImage( cvSize(img0->width/2,img0->height/2), 8, 3);
        cvResize(img0,img);
  //      src = cvCreateImage( cvSize(img0->width/2,img0->height/2), 8, 3);
  //      cvResize(img0,src);
  //      cvNamedWindow( "img0", 1 );  
  //      cvShowImage( "img0", img );
		////IplImage* imageHSV =cvCreateImage(cvGetSize(img),img->depth,3);
		////IplImage* image_11 =cvCreateImage(cvGetSize(img),img->depth,3);
		////cvCvtColor( img, image_11, CV_RGB2BGR );//BRG ת���� HSV
		////cvCvtColor( image_11, imageHSV, CV_BGR2HSV );
		////IplImage* h_space =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////IplImage* s_space =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////IplImage* v_space =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////cvSplit(imageHSV,h_space,s_space,v_space,NULL);//��ȡHSV��ͨ��
		////IplImage* h_s1 =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);//Hͨ����ֵ��
		////IplImage* h_s2 =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////IplImage* h_s3 =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////IplImage* h_s4 =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////IplImage* h_s5 =cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
		////cvThreshold(h_space,h_s1,130,180,CV_THRESH_BINARY_INV);
		////cvThreshold(h_space,h_s2,50,180,CV_THRESH_BINARY);
		////cvAnd(h_s1,h_s2,h_s3);
		////cvAnd(h_s3,h_space,h_s5);
		////IplImage* hsv =cvCreateImage(cvGetSize(img),img->depth,3);
		////IplImage* image_1 =cvCreateImage(cvGetSize(img),img->depth,3);
		////cvMerge(h_s3,h_s3,h_s3,NULL,hsv);
		////cvCvtColor( hsv,image_1, CV_HSV2BGR );
		////cvShowImage("output",image_1);
		////cvWaitKey(0);


  //      //Ϊ������ʾͼ������ռ�,3ͨ��ͼ���Ա��ò�ɫ��ʾ   			    
  //      //�����ڴ�飬���ÿ����ó�Ĭ��ֵ,��ǰĬ�ϴ�СΪ64k   
  //      CvMemStorage* storage = cvCreateMemStorage(0);  
  //      IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3);   
  //      IplImage* dst_gray = cvCreateImage( cvGetSize(src), 8, 1);   
  //      CvScalar s;
		//int uh = src->height, dh = 0, uw, dw;
		//int lw = src->width, rw = 0, lh, rh;
  //      for(int i = 0;i < src->height;i++)
  //      {
  //          for(int j = 0;j < src->width;j++)
  //          {
  //               
  //              s = cvGet2D(src,i,j); // get the (i,j) pixel value
  //              if(s.val[0] < 180 && s.val[1] < 180 && s.val[2] > 150)
  //              {
		//			uh = uh < i ? uh : i;uw = j;
		//			dh = dh > i ? dh : i;dw = j;
		//			lw = lw < j ? lw : j;lh	= i;
		//			rw = rw > j ? rw : j;rh = i;
  //                  s.val[0]=0;
  //                  s.val[1]=0;
  //                  s.val[2]=255;
  //              }
  //              else
  //              {
  //                  s.val[0]=0;
  //                  s.val[1]=0;
  //                  s.val[2]=0;
  //              }
  //              cvSet2D(src,i,j,s);   //��������
  //          }
  //      }
		////cvRectangle(src, cvPoint(lw, uh), cvPoint(rw, dh),CV_RGB(255,255,0), 2);
		//Rect lightRect(cvPoint(lw - 5, uh - 5), cvPoint(rw + 5, dh + 5));
		//cout << lw << "--" << uh << "--" <<rw << "---" << dh << endl;	
		//Mat mtx(src);
  //       cvNamedWindow( "image", 1 ); 
  //      cvShowImage("image",src);
 
 
  //      //cvCvtColor(src,dst_gray,CV_BGR2GRAY);
  //      cvCvtColor(img,dst_gray,CV_BGR2GRAY);
		////cvThreshold( dst_gray, dst_gray ,30, 255, CV_THRESH_BINARY_INV );   
		//BinarizeImageByOTSU(dst_gray);  
  //      cvNamedWindow( "��1", 1 ); 
  //      cvShowImage("��1",dst_gray);

		////        �ڶ�ֵͼ����Ѱ������ 
		////�ɶ�̬����Ԫ������   
  //      CvSeq* contour = 0;  
  //      cvFindContours( dst_gray, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE );  
		//
		////CvSeq * circles=NULL;
		////circles=cvHoughCircles(dst_gray,storage,CV_HOUGH_GRADIENT,
		////  2,   //��С�ֱ��ʣ�Ӧ��>=1
		////  15,   //�ò��������㷨���������ֵ�������ͬԲ֮�����С����
		////  200,   //����Canny�ı�Ե��ֵ���ޣ����ޱ���Ϊ���޵�һ��
		////  30,    //�ۼ����ķ�ֵ
		////  2,  //��СԲ�뾶 
		////  20  //���Բ�뾶
		////  );
		////int k;
		//// for (k=0;k<circles->total;k++)
		//// {
		////  float *p=(float*)cvGetSeqElem(circles,k);
		////  //cvCircle( img, cvPoint(cvRound(p[0]),cvRound(p[1])), 3, CV_RGB(0,255,0), -1, 8, 0 );
		////  cvCircle(img,cvPoint(cvRound(p[0]),cvRound(p[1])),cvRound(p[2]),CV_RGB(255,0,0),3,CV_AA,0);
		//// }
		////     cvNamedWindow( "img", 1 );  
  ////      cvShowImage( "img11", img );
		//// cvWaitKey(0);
  //      //cvZero( dst );//�������   
  //      //cvCvtColor(dst_gray,dst,CV_GRAY2BGR);  
  //      //Ŀ��������С����   
  //      //int mix_area = 2500;  
  //      ////Ŀ�������������   
  //      //int max_area = 3500;  
  //      //�ɴ����1-��2-��3-��4-TUPLE���͵��������ݵ�����   
  ////      CvScalar color = CV_RGB( 255, 0, 0);  
  ////      //��ͼ���л����ⲿ���ڲ������� 
		////int time = 0; 
		//multimap<double, CvSeq*> area;
  //      for(; contour != 0; contour = contour->h_next)  
  //      {
		//	CvRect rect = cvBoundingRect(contour, 0);
		//	//if(rectA_intersect_rectB(rect, lightRect))
		//	//{
		//		//++time;
		//		//CvScalar color = CV_RGB(255, 255, 0); 
		//		//CvScalar color2 = CV_RGB(0, 255, 255); 
		//		//cout << rect.x << "--" << rect.y << "--" << rect.x + rect.width << "--" << rect.y + rect.height << endl;
		//		//cvDrawContours(img, contour, CV_RGB(255, 255, 0), CV_RGB(255, 255, 0), 0, 2, CV_FILLED, cvPoint(0, 0));  
		//		//cvNamedWindow( "img", 1 ); 
		//		//cvShowImage( "img", img );
		//		//cvWaitKey(0);
  //			//} 
		//	area.insert(make_pair(cvContourArea(contour), contour));	
		//	       
		//	//cvRectangle(img, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),color,2);
  //      } 
		//multimap<double, CvSeq*>::iterator iter;
		//iter = area.begin();  
		//for(int i = 0; i < area.size() / 2; ++i)
		//	++iter;
		//double midArea = (*iter).first;
		//CvRect outRect = cvRect(img->width, img->height, 0, 0);
		//int numCircle = 0;
		//map<int, LEDCircle> ledcirclemap;
		//for(iter = area.begin(); iter != area.end(); ++iter)
		//{
		//	if((*iter).first >= (midArea * 0.7) && (*iter).first <= (midArea * 1.3))
		//	{
		//		//cvDrawContours(img, (*iter).second, CV_RGB(0, 255, 255), CV_RGB(0, 255, 255), 0, 2, CV_FILLED, cvPoint(0, 0)); 
		//		++ numCircle;
		//		CvRect rect = cvBoundingRect((*iter).second, 0);
		//		if(outRect.width == 0)
		//		{
		//			outRect.x = rect.x;
		//			outRect.y = rect.y;
		//			outRect.width = rect.width;
		//			outRect.height = rect.height;
		//		}
		//		calRect(rect, &outRect); 
 	//			
		//	}
		//}	  cvWaitKey(0);
		//cout<<numCircle;
		//if(numCircle != 64)
		//{
		//	cout << "error";
		//	getchar();
		//	return 0;
		//}
		
		//map<int, LEDCircle>::iterator iter2;
		//for(iter2 = ledcirclemap.begin(); iter2 != ledcirclemap.end(); ++iter2)
		//{
		//	(*iter2).second.drawCircle(img);
		//	cout << (*iter2).first << endl;
		//}
		////cvRectangle(img, cvPoint(outRect.x, outRect.y), cvPoint(outRect.x + outRect.width, outRect.y + outRect.height),
		//			//CV_RGB(0, 255, 0), 2);
		//cvNamedWindow( "img11", 1 );
		//cvShowImage( "img11", img );
		//cvSetImageROI(img, outRect);
		//		IplImage* newGray = cvCreateImage(cvSize(outRect.width, outRect.height), 8, 1);   
		//cvCvtColor(img,newGray,CV_BGR2GRAY);
		////cvThreshold( dst_gray, dst_gray ,30, 255, CV_THRESH_BINARY_INV );   
		//BinarizeImageByOTSU(newGray);  
  //      cvNamedWindow( "��1", 1 ); 
  //      cvShowImage("��1",newGray);

	//��ȡled������Χ����
	double outAccuracy = 0.1;
	IplImage* gray = cvCreateImage( cvSize(img->width, img->height), 8, 1);
	cvCvtColor(img, gray, CV_BGR2GRAY);
	vector<CvSeq* > outContourVector = calContour(gray, outAccuracy);
	CvRect outRect = cvRect(img->width, img->height, 0, 0);
	for(int i = 0; i != outContourVector.size(); ++ i)
	{
		CvRect rect = cvBoundingRect(outContourVector[i], 0);
		cvDrawContours(img, outContourVector[i], CV_RGB(0,255,0), CV_RGB(0,255,0), 0, 2, CV_FILLED, cvPoint(0, 0));

		if(outRect.width == 0)
		{
			outRect.x = rect.x;
			outRect.y = rect.y;
			outRect.width = rect.width;
			outRect.height = rect.height;
		}
		//cvDrawRect(img, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height), CV_RGB(200, 0, 0));
		calRect(rect, &outRect); 
	}
	cvDrawRect(img, cvPoint(outRect.x, outRect.y), cvPoint(outRect.x + outRect.width, outRect.y + outRect.height), CV_RGB(200, 0, 0));
	cvShowImage("1", img);
	cvWaitKey();

	//�led�ڲ���������
	cvSetImageROI(img, outRect);
	//IplImage* temp = cvCreateImage(cvSize(outRect.width, outRect.height), 8, 1);
	IplImage* outlineGray = cvCreateImage(cvSize(outRect.width, outRect.height), 8, 1);
	cvCvtColor(img, outlineGray, CV_BGR2GRAY);
	cvResetImageROI(img);
	//cvResize(temp, outlineGray);
	double ledAccuracy = 0.3;
	vector<CvSeq* > ledContourVector = calContour(outlineGray, ledAccuracy);
	vector<LEDCircle> ledcircleVector;
	//do
	//{
		for(int i = 0; i != ledContourVector.size(); ++ i)
		{
			LEDCircle led;
			led.calCircle(ledContourVector[i], outRect);
			led.calScalar(img);
			ledcircleVector.push_back(led);
		}
		sort(ledcircleVector.begin(), ledcircleVector.end());
		ledAccuracy += 0.1;
		for(int i = 0; i != ledcircleVector.size(); ++ i)
		{
			ledcircleVector[i].drawCircle(img);
		}
		if(ledcircleVector.size() != ledNum * ledNum)
		{
			cout <<  ledcircleVector.size();
			getchar();
		}
	//}while(ledcircleVector.size() != ledNum * ledNum);

		//        �ڶ�ֵͼ����Ѱ������ 
		//�ɶ�̬����Ԫ������   
  //      CvSeq* newContour = 0;  
  //      cvFindContours( newGray, storage, &newContour, sizeof(CvContour), CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE );
		//for(; newContour != 0; newContour = newContour->h_next)  
  //      { 	
		//	LEDCircle led;
		//	led.calCircle((*iter).second);
		//	led.drawCircle(img);
		//	cvNamedWindow( "img11", 1 );
		//	cvShowImage( "img11", img );
		//	ledcirclemap.insert(make_pair(numCircle, led));

		//	//CvRect rect = cvBoundingRect(newContour, 0);
		//	//if(rectA_intersect_rectB(rect, lightRect))
		//	//{
		//		//++time;
		//		//CvScalar color = CV_RGB(255, 255, 0); 
		//		//CvScalar color2 = CV_RGB(0, 255, 255); 
		//		//cout << rect.x << "--" << rect.y << "--" << rect.x + rect.width << "--" << rect.y + rect.height << endl;
		//		//cvDrawContours(img, newContour, CV_RGB(255, 255, 0), CV_RGB(255, 255, 0), 0, 2, CV_FILLED, cvPoint(0, 0));  
  //			//} 
		//	//area.insert(make_pair(cvContourArea(newContour), newContour));	
		//	       
		//	//cvRectangle(img, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),color,2);
  //      } 
						cvNamedWindow( "img", 1 ); 
				cvShowImage( "img", img );

 		//cvSmooth( dst_gray, dst_gray, CV_GAUSSIAN, 5, 5 ); 		
		//CvMemStorage* sto=cvCreateMemStorage(0);
		//CvSeq* results=cvHoughCircles(dst_gray,sto,CV_HOUGH_GRADIENT,1, 18, 100, 20, 32, 48); // 
		//for(int i=0;i<results->total;i++)								 	
		//{
		//	float* p=(float*)cvGetSeqElem(results,i);
		//	CvPoint pt=cvPoint(cvRound(p[0]),cvRound(p[1]));
		//	cvCircle(img,pt,p[2],CV_RGB(255,0,0), 5);
		//}
		//cvNamedWindow("TestHoughCircles",1);
		//cvShowImage("TestHoughCircles",img);
        //  //��ͼ��������ʶ�ֵ��   
        ////BinarizeImageByOTSU(dst_gray);  
        // 
        ////ͼ������   
        //cvDilate(dst_gray,dst_gray);  
        ////ͼ��ʴ   
        //cvErode(dst_gray,dst_gray);
        ////��ʾԴͼ��Ķ�ֵͼ   
        //cvNamedWindow( "Source", 1 );  
        //cvShowImage( "Source", dst_gray );  
        //�ڶ�ֵͼ����Ѱ������   
  //      cvFindContours( dst_gray, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL , CV_CHAIN_APPROX_SIMPLE );  
  //      //cvZero( dst );//�������   
  //      //cvCvtColor(dst_gray,dst,CV_GRAY2BGR);  
  //      //Ŀ��������С����   
  //      //int mix_area = 2500;  
  //      ////Ŀ�������������   
  //      //int max_area = 3500;  
  //      //�ɴ����1-��2-��3-��4-TUPLE���͵��������ݵ�����   
  //      CvScalar color = CV_RGB( 255, 0, 0);  
  //      //��ͼ���л����ⲿ���ڲ������� 
		//int i = 0;  
  //      for(; contour != 0; contour = contour->h_next)  
  //      {  
		//	++i;
		//	CvScalar color = CV_RGB(255, 255, 0); 
		//	CvScalar color2 = CV_RGB(0, 255, 255); 
		//	cvDrawContours(img, contour, color2, color, 0, 2, CV_FILLED, cvPoint(0, 0));  
		//	//CvRect rect = cvBoundingRect(contour, 0); 
		//	//cvRectangle(img, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height),color,2);
  //      } 
		//cout << i;
  //      cvNamedWindow( "img", 1 );  
  //      cvShowImage( "img", img );

        cvWaitKey(0); 
        //cvDestroyWindow("img0");
        cvDestroyWindow("img");
        //cvDestroyWindow("�Ҷ�ͼ");
        //cvReleaseImage(&dst);  
        //cvDestroyWindow("Source");  
 
        //cvReleaseImage(&dst_gray);
        //cvReleaseImage(&src);
        cvReleaseImage(&img);
        //cvReleaseImage(&img0);
         
        return 0;  
    }     
    return 1;  
}