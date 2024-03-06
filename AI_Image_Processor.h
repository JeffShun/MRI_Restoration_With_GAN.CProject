
#include <vector>
#include <stack>
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

struct ItkData {
	string dcmPath;
	ImageType2F::Pointer img;
	ImageType2F::SizeType size;
	itk::MetaDataDictionary meta_dict;
};

class ImageProcessor {
public:
	ImageProcessor(int mode);
	~ImageProcessor();
	void pushToStack(string dataPath);
	string popFromStack();
	void processStart();

private:
	ICudaEngine* engine;
	Logger logger;
	stack<string> dataStack;
	int mode;

	ItkData readImage(string dcmPath);
	ItkData preprocessImage(ItkData inputData);
	void printMetaData(const itk::MetaDataDictionary& meta_dict);

	void resizeImg(
		ImageType2F::Pointer oriImg,
		ImageType2F::Pointer& outImg,
		ImageType2F::SizeType targetSize);

	ICudaEngine* buildEngine(
		string engine_model_path,
		string onnx_model_path,
		ILogger& logger);

	void enginePredict(ICudaEngine *engine, ItkData& inputData);
	void postProcess(ItkData& inputData, ImageType2F::SizeType oShape);
	void saveAsDICOM(ItkData inputData);
};