#include "filename.h"
#include "qgpsequentialbatch.h"
#include "qgpclassification.h"

QGPSequentialBatch::QGPSequentialBatch() :QGPBatch(){

}

QGPSequentialBatch::QGPSequentialBatch(Module *module) : QGPBatch(module){

}

QGPSequentialBatch::~QGPSequentialBatch(){
    if(this->nonzero != nullptr){
        delete[] this->nonzero;
    }
}

void QGPSequentialBatch::load(){
    //qDebug() << "QGPBatch::load range[" << fromIndex << "," << toIndex << ")";
    QGPBatch::LData data;
	if(this->nonzero != nullptr){
        delete[] this->nonzero;
	}
	
    this->nonzero = new bool[224000];
    std::fill(&this->nonzero[0],&this->nonzero[224000],false);
    std::string filename;
    for (size_t k = fromIndex; k < toIndex; ++k) {
        filename = module->filename.getAbsoluteFilename(k, QGPClassification::QGP);
        data = LData::fromFile(filename, nonzero);
        data.classification = QGPClassification::QGP;
        this->data.emplace_back(data);

        filename = module->filename.getAbsoluteFilename(k,QGPClassification::NonQGP);
        data = LData::fromFile(filename, nonzero);
        data.classification = QGPClassification::NonQGP;
        this->data.emplace_back(data);
    }
}
