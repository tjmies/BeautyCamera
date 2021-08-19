#pragma once
#include "define.h"
class BeautyCam
{
public:
	BeautyCam();
	void initMainImgUI();
	static void browersBut_callback(int state, void *data);
	//�Աȶȵ���
	static void on_contrast(int b, void*userdata);
	//���ȵ���
	static void on_lightness(int b, void*userdata);
	//�۾�����
	static void on_BigEye(int b, void*userdata);
	//����Ч��
	static void on_thinFace(int b, void*userdata);
	//����Ч��
	static void on_beautyFace(int b, void*userdata);
	//��ȡ�����ؼ���
	std::vector<std::vector<Point2f>> dectectFace68(const string &path);
	//�ֲ�ƽ���۲��Ŵ�
	void LocalTranslationWarp_Eye(Mat &img, Mat &dst, int warpX, int warpY, int endX, int endY, float radius);
	//���������Բ�ֵ
	void BilinearInsert(Mat &src, Mat &dst, float ux, float uy, int i, int j);
	//�ֲ�ƽ������
	Mat LocalTranslationWarp_Face(Mat &img, int warpX, int warpY, int endX, int endY, float radius);
private:
	//���ؼ���������
	CascadeClassifier loadCascadeClassifier(const string cascadePath);
	// ���ͻ���
	void detectAndDraw(Mat& img,CascadeClassifier& cascade,double scale,int val);
private:
	Mat m_MainImg;
	static BeautyCam *m_pIntance; 
	std::vector<std::vector<Point2f>> m_vecFaceData;
	bool isDetected = false;
};

