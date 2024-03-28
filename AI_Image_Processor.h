
#include <vector>
#include <queue>
#include <string>
#include "itkImage.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;

typedef itk::Image<float, 2> ImageType2F;

class Logger : public ILogger
{
	virtual void log(Severity severity, const char* msg) noexcept override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			cout << msg << endl;
	}
} gLogger;


struct mData {
	string inputPath;
	float sr_ratio;
	float deblur_ratio;
	float denoise_ratio;
	ImageType2F::Pointer inImg;
	ImageType2F::SizeType inSize;
	itk::MetaDataDictionary inMetaDict;
	ImageType2F::Pointer outImg;
	ImageType2F::SizeType outSize;
	itk::MetaDataDictionary outMetaDict;
};


class ImageProcessor {
public:
	ImageProcessor(string engine_model_path, string onnx_model_path);
	~ImageProcessor();

	mData dataStream;

	void pushToQueue(mData& dataStream);
	void processStart();
	size_t queueLen();

	float sr_ratio;
	float deblur_ratio;
	float denoise_ratio;

private:
	ICudaEngine* engine;
	Logger logger;
	queue<mData> dataQueue;
	string engine_model_path;
	string onnx_model_path;

	void readImage(mData& dataStream);
	void preprocessImage(mData& dataStream);
	void enginePredict(mData& dataStream);
	void postProcess(mData& dataStream);
	void saveDICOM(mData& dataStream);

	void resizeImg(
		ImageType2F::Pointer oriImg,
		ImageType2F::Pointer& outImg,
		ImageType2F::SizeType targetSize);

	ICudaEngine* buildEngine(
		string engine_model_path,
		string onnx_model_path,
		ILogger& logger);
};