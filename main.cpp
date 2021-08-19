#include "define.h"
//#include "OpenCVOperation.h"
#include "BeautyCam.h"
void test()
{

	Mat A = imread("beatiful.jpg");
	Mat ROI = A(Rect(0, 0, 200, 200));

	Mat B = imread("right-eye.jpg");

	//ROI=B;	ǳ�㿽��1��ROI�ᱻ�ı䣬��Aû�б��ı�
	//ROI(B);	ǳ�㿽��2����������붼�����
	resize(B, B, Size(200, 200));
	B.copyTo(ROI);		//��㿽��copyTo()��ROI�ᱻ�ı䣬����AҲ���ı���	
	//ROI=B.clone();			//��㿽��clone()��ROI�ᱻ�ı䣬��Aû�б��ı�
	imshow("A-changed", A);
	imshow("ROI changed", ROI);


}
void checkAddressFormat(string from, string name)
{

	string str = "0xceeef35bebdf1876d55d16adfa28b79c368d92e4";
	if (_stricmp(from.c_str(), str.c_str()) == 0)
	{
		cout << "yiyang"+name<<endl;
	}
}
int main(int argc, const char** argv)
{
	//OpenCVOperation op;

	//op.beautyCamera();
	//op.beautyPicture();
	//op.colorTransform("beatiful.jpg");

	Mat img = imread("33.jpg");
	if (img.empty())
	{
		cout << "img empty" << endl;
		return 0;
	}
	//op.mouse_draw_demo(img);
	//imshow("����ͼ��", img);
	//Mat dst = Mat::zeros(Size(img.cols*1.5, img.rows*1.5), img.type());
	
	//op.zhifangtu2Demo(img);
	//op.handlevideo("111.mp4");
	//op.zoom(img, dst);
	//op.thin_face_dlib("sss.jpg");
	//op.thin_eyes_dlib("sss.jpg");
	//op.drawing_demo(img);
	//op.enlargeEyes(img);
	//op.pixel_statistic_demo(img);

	//op.pixel_visit_demo(img);
	//op.tracking_bar_demo(img);
	//op.key_demo(img);
	//op.color_style_demo(img);
	//op.bitwise_demo(img);
	//op.inrange_demo(img);


	BeautyCam cam;







	

	waitKey(0);
	cv::destroyAllWindows();

	return 0;

}