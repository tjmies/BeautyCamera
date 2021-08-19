#include "BeautyCam.h"
BeautyCam *BeautyCam::m_pIntance=nullptr;

BeautyCam::BeautyCam()
{
	//初始化静态对象
	m_pIntance = this;
	//初始化主窗口
	initMainImgUI();
}
void BeautyCam::initMainImgUI()
{
	string path;
	cout << "请输入图片的绝对路径：" << endl;
	cin>> path;
	
	namedWindow("BeautyCam", WINDOW_AUTOSIZE);
	//string path = "33.jpg";


	m_MainImg =imread(path);
	imshow("src", m_MainImg);

	//检测人脸数据68点
	m_vecFaceData = dectectFace68(path);

	int max_value = 100;
	int con_value = 100;
	int lignhtnesss = 50;
	int contrast = 50;
	int bigeyeval = 0;
	int faceval = 0;
	int beautyval = 0;
	createTrackbar("亮度", "BeautyCam", &lignhtnesss, max_value, on_lightness, (void*)(&m_MainImg));
	createTrackbar("对比度", "BeautyCam", &contrast, max_value, on_contrast, (void*)(&m_MainImg));
	createTrackbar("大眼", "BeautyCam", &bigeyeval, 60, on_BigEye, (void*)(&m_MainImg));
	createTrackbar("瘦脸", "BeautyCam", &faceval, 70, on_thinFace, (void*)(&m_MainImg));
	createTrackbar("美颜", "BeautyCam", &beautyval, 200, on_beautyFace, (void*)(&m_MainImg));

	on_lightness(50, (void*)(&m_MainImg));
	//imshow("BeautyCam", m_MainImg);
}
void BeautyCam::browersBut_callback(int state, void *data)
{
	cout << "state" << state << endl;
}
void BeautyCam::on_lightness(int b, void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	m = Scalar(b, b, b);
	//add(img, m, dst);
	addWeighted(img, 1.0, m, 0, b, dst);
	imshow("BeautyCam", dst);
}
void BeautyCam::on_contrast(int b, void*userdata)
{
	Mat img = *((Mat *)userdata);
	Mat m = Mat::zeros(img.size(), img.type());
	Mat dst = Mat::zeros(img.size(), img.type());
	m = Scalar(b, b, b);
	double con = b / 100.0;
	addWeighted(img, con, m, 0, 0, dst);
	imshow("BeautyCam", dst);
}
void BeautyCam::on_BigEye(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat dst = src.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f left_landmark = points_vec[38];
		Point2f	left_landmark_down = points_vec[27];

		Point2f	right_landmark = points_vec[44];
		Point2f	right_landmark_down = points_vec[27];

		Point2f	endPt = points_vec[30];

		//# 计算第4个点到第6个点的距离作为距离
		/*float r_left = sqrt(
			(left_landmark.x - left_landmark_down.x) * (left_landmark.x - left_landmark_down.x) +
			(left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));
		cout << "左眼距离:" << r_left;*/
		float r_left = b;

		//	# 计算第14个点到第16个点的距离作为距离
		//float	r_right = sqrt(
		//	(right_landmark.x - right_landmark_down.x) * (right_landmark.x - right_landmark_down.x) +
		//	(right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y));
		//cout << "右眼距离:" << r_right;
		float r_right = b;
		//	# 瘦左                     
		m_pIntance->LocalTranslationWarp_Eye(src, dst, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
		//	# 瘦右
		m_pIntance->LocalTranslationWarp_Eye(src, dst, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);

	}
	imshow("BeautyCam", dst);
}
std::vector<std::vector<Point2f>> BeautyCam::dectectFace68(const string &path)
{
	std::vector<std::vector<Point2f>>  rets;
	//加载图片路径
	array2d<rgb_pixel> img;
	load_image(img, path.c_str());
	//定义人脸检测器
	frontal_face_detector detector = get_frontal_face_detector();
	std::vector<dlib::rectangle> dets = detector(img);

	for (auto var : dets)
	{
		//关键点检测器
		shape_predictor sp;
		deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
		//定义shape对象保存检测的68个关键点
		full_object_detection shape = sp(img, var);
		//存储文件
		ofstream out("face_detector.txt");
		//读取关键点到容器中
		std::vector<Point2f> points_vec;
		for (int i = 0; i < shape.num_parts(); ++i)
		{
			auto a = shape.part(i);
			out << a.x() << " " << a.y() << " ";
			Point2f ff(a.x(), a.y());
			points_vec.push_back(ff);
		}
		rets.push_back(points_vec);
	}
	cout << "人脸检测结束:" <<dets.size()<<"张人脸数据"<< endl;
	return rets;
}
void BeautyCam::LocalTranslationWarp_Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius)
{
	//平移距离 
	float ddradius = radius * radius;
	//计算|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//计算 图像的高  宽 通道数量
	int height = img.rows;
	int width = img.cols;
	int chan = img.channels();

	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// # 计算该点是否在形变圆的范围之内
			//# 优化，第一步，直接判断是会在（startX, startY)的矩阵框中
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				float rnorm = sqrt(distance) / radius;
				float ratio = 1 - (rnorm - 1)*(rnorm - 1)*0.5;
				//映射原位置
				float UX = warpX + ratio * (i - warpX);
				float UY = warpY + ratio * (j - warpY);

				//根据双线性插值得到UX UY的值
				BilinearInsert(img, dst, UX, UY, i, j);
			}
		}
	}

}
void BeautyCam::BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j)
{
	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	int c = src.channels();
	if (c == 3)
	{
		//存储图像得浮点坐标
		CvPoint2D32f uv;
		CvPoint3D32f f1;
		CvPoint3D32f f2;

		//取整数
		int iu = (int)ux;
		int iv = (int)uy;
		uv.x = iu + 1;
		uv.y = iv + 1;

		//step图象像素行的实际宽度  三个通道进行计算(0 , 1 2  三通道)
		f1.x = ((uchar*)(src.data + src.step*iv))[iu * 3] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3] * (uv.x - iu);
		f1.y = ((uchar*)(src.data + src.step*iv))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 1] * (uv.x - iu);
		f1.z = ((uchar*)(src.data + src.step*iv))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*iv))[(iu + 1) * 3 + 2] * (uv.x - iu);


		f2.x = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3] * (uv.x - iu);
		f2.y = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 1] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 1] * (uv.x - iu);
		f2.z = ((uchar*)(src.data + src.step*(iv + 1)))[iu * 3 + 2] * (1 - Abs(uv.x - iu)) + \
			((uchar*)(src.data + src.step*(iv + 1)))[(iu + 1) * 3 + 2] * (uv.x - iu);

		((uchar*)(dst.data + dst.step*j))[i * 3] = f1.x*(1 - Abs(uv.y - iv)) + f2.x*(Abs(uv.y - iv));  //三个通道进行赋值
		((uchar*)(dst.data + dst.step*j))[i * 3 + 1] = f1.y*(1 - Abs(uv.y - iv)) + f2.y*(Abs(uv.y - iv));
		((uchar*)(dst.data + dst.step*j))[i * 3 + 2] = f1.z*(1 - Abs(uv.y - iv)) + f2.z*(Abs(uv.y - iv));

	}
}
void BeautyCam::on_thinFace(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat dst = src.clone();
	for (auto points_vec : m_pIntance->m_vecFaceData)
	{
		Point2f endPt = points_vec[34];
		for (int i = 3; i < 15; i = i + 2)
		{
			Point2f start_landmark = points_vec[i];
			Point2f end_landmark = points_vec[i + 2];

			//计算瘦脸距离
			/*float dis = sqrt(
				(start_landmark.x - end_landmark.x) * (start_landmark.x - end_landmark.x) +
				(start_landmark.y - end_landmark.y) * (start_landmark.y - end_landmark.y));*/
			float dis = b;
			dst = m_pIntance->LocalTranslationWarp_Face(dst, start_landmark.x, start_landmark.y, endPt.x, endPt.y, dis);

			/*
			//指定位置
			Point2f left_landmark = points_vec[2];
			Point2f	left_landmark_down = points_vec[5];

			Point2f	right_landmark = points_vec[13];
			Point2f	right_landmark_down = points_vec[15];

			Point2f	endPt = points_vec[30];

			//# 计算第4个点到第6个点的距离作为瘦脸距离
			float r_left = sqrt(
				(left_landmark.x - left_landmark_down.x) * (left_landmark.x - left_landmark_down.x) +
				(left_landmark.y - left_landmark_down.y) * (left_landmark.y - left_landmark_down.y));
			cout << "左边瘦脸距离:" << r_left;


			//	# 计算第14个点到第16个点的距离作为瘦脸距离
			float	r_right = sqrt(
				(right_landmark.x - right_landmark_down.x) * (right_landmark.x - right_landmark_down.x) +
				(right_landmark.y - right_landmark_down.y) * (right_landmark.y - right_landmark_down.y));
			cout << "右边瘦脸距离:" << r_right;
				//	# 瘦左边脸                         源图像   坐标移动点 x           y                  结束点x       y     瘦脸距离
			LocalTranslationWarp(src, dst, left_landmark.x, left_landmark.y, endPt.x, endPt.y, r_left);
			//	# 瘦右边脸
			LocalTranslationWarp(src, dst, right_landmark.x, right_landmark.y, endPt.x, endPt.y, r_right);
			*/
		}
	}
	imshow("BeautyCam", dst);
}
Mat BeautyCam::LocalTranslationWarp_Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius)
{
	Mat dst = img.clone();
	//平移距离 
	float ddradius = radius * radius;
	//计算|m-c|^2
	size_t mc = (endX - warpX)*(endX - warpX) + (endY - warpY)*(endY - warpY);
	//计算 图像的高  宽 通道数量
	int height = img.rows;
	int width = img.cols;
	int chan = img.channels();

	auto Abs = [&](float f) {
		return f > 0 ? f : -f;
	};

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			// # 计算该点是否在形变圆的范围之内
			//# 优化，第一步，直接判断是会在（startX, startY)的矩阵框中
			if ((Abs(i - warpX) > radius) && (Abs(j - warpY) > radius))
				continue;

			float distance = (i - warpX)*(i - warpX) + (j - warpY)*(j - warpY);
			if (distance < ddradius)
			{
				//# 计算出（i, j）坐标的原坐标
				//# 计算公式中右边平方号里的部分
				float ratio = (ddradius - distance) / (ddradius - distance + mc);
				ratio *= ratio;

				//映射原位置
				float UX = i - ratio * (endX - warpX);
				float UY = j - ratio * (endY - warpY);

				//根据双线性插值得到UX UY的值
				BilinearInsert(img, dst, UX, UY, i, j);
				//改变当前的值
			}
		}
	}

	return dst;

}
void BeautyCam::on_beautyFace(int b, void*userdata)
{
	Mat src = *((Mat *)userdata);
	Mat img = src.clone();
	double scale = 1.3;
	
	CascadeClassifier cascade = m_pIntance->loadCascadeClassifier("./haarcascade_frontalface_alt.xml");//人脸的训练数据
	CascadeClassifier netcascade = m_pIntance->loadCascadeClassifier("./haarcascade_eye_tree_eyeglasses.xml");//人眼的训练数据
	if (cascade.empty() || netcascade.empty())
		return;
	m_pIntance->detectAndDraw(img, cascade, scale,b);
	if (m_pIntance->isDetected == false)
	{
		cout << "enter" << endl;
		Mat dst;

		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //双边滤波参数之一  
		//double fc = value1 * 12.5; //双边滤波参数之一  
		double fc = b;
		int p = 50;//透明度  
		Mat temp1, temp2, temp3, temp4;

		//对原图层image进行双边滤波，结果存入temp1图层中
		bilateralFilter(img, temp1, dx, fc, fc);

		//将temp1图层减去原图层image，将结果存入temp2图层中
		temp2 = (temp1 - img + 128);

		//高斯模糊  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);

		//以原图层image为基色，以temp3图层为混合色，将两个图层进行线性光混合得到图层temp4
		temp4 = img + 2 * temp3 - 255;

		//考虑不透明度，修正上一步的结果，得到最终图像dst
		dst = (img*(100 - p) + temp4 * p) / 100;
		dst.copyTo(img);
	}
	imshow("BeautyCam", img);
}
CascadeClassifier BeautyCam::loadCascadeClassifier(const string cascadePath)
{
	CascadeClassifier cascade;
	if (!cascadePath.empty())
	{
		if(!cascade.load(cascadePath))//从指定的文件目录中加载级联分类器
		{
			cerr << "ERROR: Could not load classifier cascade" << endl;
		}
	}
	return cascade;
}
void BeautyCam::detectAndDraw(Mat& img, CascadeClassifier& cascade,  double scale, int val)
{
	std::vector<Rect> faces;
	const static Scalar colors[] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };//用不同的颜色表示不同的人脸
	//将图片缩小，加快检测速度
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	//因为用的是类haar特征，所以都是基于灰度图像的，这里要转换成灰度图像
	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//将尺寸缩小到1/scale,用线性插值
	equalizeHist(smallImg, smallImg);//直方图均衡
	cascade.detectMultiScale(smallImg, //image表示的是要检测的输入图像
		faces,//objects表示检测到的人脸目标序列
		1.1, //caleFactor表示每次图像尺寸减小的比例
		2, //minNeighbors表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
		0 | CASCADE_SCALE_IMAGE ,//minSize为目标的最小尺寸
		Size(30, 30)); //minSize为目标的最大尺寸
	int i = 0;
	//遍历检测的矩形框
	for (std::vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		isDetected = true;
		Mat smallImgROI;
		std::vector<Rect> nestedObjects;
		Point center, left, right;
		Scalar color = colors[i % 8];
		int radius;
		center.x = cvRound((r->x + r->width*0.5)*scale);//还原成原来的大小
		center.y = cvRound((r->y + r->height*0.5)*scale);
		radius = cvRound((r->width + r->height)*0.25*scale);

		left.x = center.x - radius;
		left.y = cvRound(center.y - radius * 1.3);

		if (left.y < 0)
		{
			left.y = 0;
		}
		right.x = center.x + radius;
		right.y = cvRound(center.y + radius * 1.3);

		if (right.y > img.rows)
		{
			right.y = img.rows;
		}
		/*原理算法
		美肤-磨皮算法
		Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
		*/
		//绘画识别的人脸框
		//rectangle(img, left, right, Scalar(255, 0, 0));
		Mat roi = img(Range(left.y, right.y), Range(left.x, right.x));
		
		Mat dst;
		int value1 = 3, value2 = 1;

		int dx = value1 * 5;    //双边滤波参数之一  
		//double fc = value1 * 12.5; //双边滤波参数之一 
		double fc = val;//变化值
		int p = 50;//透明度  
		Mat temp1, temp2, temp3, temp4;

		//双边滤波    输入图像 输出图像 每像素领域的直径范围颜色空间过滤器的sigma  坐标空间滤波器的sigma 
		bilateralFilter(roi, temp1, dx, fc, fc);
		temp2 = (temp1 - roi + 128);
		//高斯模糊  
		GaussianBlur(temp2, temp3, Size(2 * value2 - 1, 2 * value2 - 1), 0, 0);
		temp4 = roi + 2 * temp3 - 255;
		dst = (roi*(100 - p) + temp4 * p) / 100;
		dst.copyTo(roi);
	}
}