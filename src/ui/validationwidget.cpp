#include "validationwidget.h"
#include "ui_validationwidget.h"
#include <fstream>

ValidationWidget::ValidationWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ValidationWidget),
    supervisor(new Supervisor())
{
    ui->setupUi(this);

    std::ifstream lastFilePath("lastFilePath");
    if (!lastFilePath.fail()) {
        std::string path;
        getline(lastFilePath, path);
        qDebug() << path.c_str();
        //supervisor->fDirname = path; todo module
    }

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &ValidationWidget::renew);



    connect(ui->btn_analyze, SIGNAL(clicked()), this,
            SLOT(analyzeQGP()));  // Start DIRECT MODE with press the button "Direct Mode"
    connect(ui->sb_szTraining, SIGNAL(valueChanged(int)), this, SLOT(setTrain(int)));     // choose TRAIN MODE number
    connect(ui->sb_szValidation, SIGNAL(valueChanged(int)), this, SLOT(setTest(int)));     // choose TEST MODE number
    connect(ui->sb_szBatches, SIGNAL(valueChanged(int)), this, SLOT(setBatch(int)));

    connect(ui->btn_loadNet,SIGNAL(clicked()),this,SLOT(loadNet()));
    connect(ui->btn_saveNet,SIGNAL(clicked()),this,SLOT(saveNet()));

    connect(ui->btn_loadData, SIGNAL(clicked()), this, SLOT(loadQGP()));   //Load QGP and NQGP Data


    ui->bigChart->clearGraphs();
    ui->bigChart->addGraph();
    ui->bigChart->xAxis->setLabel("NEpoch");
    ui->bigChart->yAxis->setLabel("Efficiency");
    ui->bigChart->xAxis->setRange(0, szEpoch + 1);
    ui->bigChart->yAxis->setRange(70, 105);
    ui->bigChart->replot();


    ui->smallChart->clearGraphs();
    ui->smallChart->addGraph();
    ui->smallChart->xAxis->setLabel("NEpoch");
    ui->smallChart->yAxis->setLabel("Loss");
    ui->smallChart->xAxis->setRange(0, szEpoch + 1);
    ui->smallChart->yAxis->setRange(0, 1);
    ui->smallChart->replot();


    supervisor->szTraining = size_t(ui->sb_szTraining->value());
    supervisor->szValidation = size_t(ui->sb_szValidation->value());
    supervisor->setBatchSize(ui->sb_szBatches->value());
}

ValidationWidget::~ValidationWidget(){
    delete ui;
    delete ui;
    if(this->process != nullptr) {
        delete this->process;
    }
}

void ValidationWidget::loadNeuralNet()
{
    if(this->process == nullptr){
    QString fileName = QFileDialog::getOpenFileName(this,
           tr("Open Neural Net"), "",
           tr("NeuralNet (*.nn);;All Files (*)"));

    if(fileName == ""){
        return;
    }
    delete this->supervisor->neuralNet;
    std::string str = fileName.toStdString();
    this->supervisor->neuralNet = NeuralNet::fromFile(str);
    }
}

void ValidationWidget::saveNeuralNet()
{
    if(this->process == nullptr){
        QFileDialog *dialog = new QFileDialog();
        dialog->setDefaultSuffix("nn");
    QString fileName = dialog->getSaveFileName(this,
           tr("Save Neural Net"), "",
           tr("NeuralNet (*.nn);;All Files (*)"));

    qDebug() << fileName;
    QString fileNameExt =fileName;
    QString statfilename = fileName + ".stats";
    this->supervisor->neuralNet->toFile(fileNameExt.toStdString());
    this->supervisor->toFile(statfilename.toStdString());
    delete dialog;
    }
}

void ValidationWidget::loadDataset()
{
    QString fDirname = QFileDialog::getExistingDirectory(this, tr("Open input data directory"), QDir::homePath());
    //this->supervisor->fDirname = fDirname.toUtf8().constData(); // todo stdString
    // todo use module
    qDebug() << "Loading from filepath: fDirname: " << fDirname;
    std::ofstream lastFilePath("lastFilePath");

    if (!lastFilePath.fail()) {
        lastFilePath << fDirname.toUtf8().constData();
        lastFilePath.close();
    }
}

void ValidationWidget::analyzeData()
{
    if (this->supervisor->running) {
        this->supervisor->running = false; // todo implementation for stop / pause
    } else {
        supervisor->szEpoch = 1;
        this->ui->bigChart->clearGraphs();
        this->ui->smallChart->clearGraphs();

        this->ui->sb_szBatches->setEnabled(false);
        this->ui->sb_szTraining->setEnabled(false);
        this->ui->sb_szValidation->setEnabled(false);
        this->ui->btn_loadData->setEnabled(false);


        this->ui->pro_Training->setMaximum(supervisor->szTraining * 2 + supervisor->szValidation * 2);
        this->ui->pro_Epoch->setMaximum(supervisor->szEpoch);
        this->process = new std::thread(&Supervisor::runValidation, supervisor);
        timer->start(5000);
        this->ui->btn_analyze->setText("Pause");
    }
}

void ValidationWidget::setOffset(int number)
{

}

void ValidationWidget::setTestdata(int number)
{

}

void ValidationWidget::setBatchsize(int number)
{
      supervisor->setBatchSize(size_t(number));

      szEpoch = size_t(number);

      supervisor->setEpoch(szEpoch);

      ui->bigChart->xAxis->setRange(0, szEpoch + 1);
      ui->bigChart->replot();

      ui->smallChart->xAxis->setRange(0, szEpoch + 1);
      ui->smallChart->replot();
}

void ValidationWidget::drawHistogram(size_t iEp)
{
    const std::vector <size_t> &fRunEventsA = supervisor->statTotalTraining;
    const std::vector <size_t> &fRunEventsT = supervisor->statTotalValidation;

    const std::vector <size_t> &fGoodEventsA = supervisor->statCorrectTraining;
    const std::vector <size_t> &fGoodEventsT = supervisor->statCorrectValidation;

    ui->bigChart->clearGraphs();
    ui->bigChart->addGraph();
    ui->bigChart->xAxis->setLabel("NEpoch");
    ui->bigChart->yAxis->setLabel("Efficiency");
    ui->bigChart->xAxis->setRange(0, szEpoch + 1);
    ui->bigChart->yAxis->setRange(70, 105);
    ui->bigChart->replot();

    QVector<double> xTraining(iEp + 2);
    QVector<double> yTraining(iEp + 2);

    bool last = supervisor->curValidation != 0;
    QVector<double> yValidation;
    QVector<double> xValidation;
    if(last){
        xValidation = QVector<double>(iEp+2);
        yValidation = QVector<double>(iEp + 2);
    } else {
        xValidation = QVector<double>(iEp+1);
        yValidation = QVector<double>(iEp + 1);
    }

    for (size_t i = 0; i < iEp + 1; i++) {
        xTraining[i] = i;
        xValidation[i] = i;
    }
    xTraining[iEp+1] = iEp+1;
    if(last){
        xValidation[iEp +1] = iEp+1;
    }
    yTraining[0] = 0;
    yValidation[0] = 0;
    for (size_t i = 1; i < iEp + 1; i++) {
        if (fRunEventsA[size_t(i - 1)]) yTraining[i] = 100 * number(fGoodEventsA[size_t(i - 1)]) / fRunEventsA[size_t(i - 1)];
        if (fRunEventsT[size_t(i - 1)]) {
            if (this->supervisor->szValidation) {
                yValidation[i] = 100 * number(fGoodEventsT[size_t(i - 1)]) / fRunEventsT[size_t(i - 1)];
            } else {
                yValidation[i] = 0.0;
            }
        }
    }

    if (fRunEventsA[size_t(iEp)]) {
        yTraining[iEp + 1] = 100 * number(fGoodEventsA[size_t(iEp)]) / fRunEventsA[size_t(iEp)];
    }
    if (last && fRunEventsT[size_t(iEp)]) {
        if (this->supervisor->szValidation) {
            yValidation[iEp + 1] = 100 * number(fGoodEventsT[size_t(iEp + 1 - 1)]) / fRunEventsT[size_t(iEp + 1 - 1)];
        }        else {
            yValidation[iEp + 1] = 0;
        }
    }
    ui->bigChart->addGraph();
    ui->bigChart->addGraph();
    ui->bigChart->graph(0)->setData(xTraining, yTraining);
    ui->bigChart->graph(1)->setData(xValidation, yValidation);

    ui->bigChart->graph(0)->setScatterStyle(QCPScatterStyle::ssSquare);
    ui->bigChart->graph(1)->setScatterStyle(QCPScatterStyle::ssSquare);

    QPen pen;
    pen.setColor(QColor(125, 125, 225));
    pen.setWidth(2);
    ui->bigChart->graph(1)->setPen(pen);

    pen.setColor(QColor(225, 125, 125));
    pen.setStyle(Qt::DashLine);
    ui->bigChart->graph(0)->setPen(pen);

    ui->bigChart->replot();
}

void ValidationWidget::drawLoss(size_t iEp)
{
    const std::vector<number> &fRunEpochsL = supervisor->statLossTraining;
    const std::vector<number> &fValidateEpochsL = supervisor->statLossValidation;

    QVector<double> xTraining(iEp + 2);
    QVector<double> yTraining(iEp + 2);

    QVector<double> xValidation;
    QVector<double> yValidation;

    bool last = this->supervisor->curValidation != 0;
    if(last){
        xValidation = QVector<double>(iEp + 2);
        yValidation = QVector<double>(iEp + 2);
    } else {
        xValidation = QVector<double>(iEp + 1);
        yValidation = QVector<double>(iEp + 1);
    }

    for (size_t i = 0; i < iEp + 1; i++) {
        xTraining[i] = i;
        xValidation[i] = i;
    }
    xTraining[iEp+1] = iEp+1;
    if(last){
        xValidation[iEp+1] = iEp+1;
    }

    yTraining[0] = 0;
    yValidation[0] = 0;
    for (size_t i = 1; i < iEp + 1; i++) {
        yTraining[i] = fRunEpochsL[i - 1];
        yValidation[i] = fValidateEpochsL[i - 1];
    }
    yTraining[iEp + 1] = fRunEpochsL[iEp + 1 - 1];
    if(last){
        yValidation[iEp + 1] = fValidateEpochsL[iEp +1 - 1];
    }


    ui->smallChart->clearGraphs();
    ui->smallChart->addGraph();
    ui->smallChart->graph(0)->setData(xTraining, yTraining);

    ui->smallChart->addGraph();
    ui->smallChart->graph(1)->setData(xValidation, yValidation);

    QPen pen;
    // pen.setColor(QColor(25, 135, 25));
    pen.setColor(QColor(125, 125, 225));
    pen.setWidth(2);
    ui->smallChart->graph(1)->setPen(pen);

    pen.setColor(QColor(225, 125, 125));
    pen.setWidth(2);
    pen.setStyle(Qt::DashLine);
    ui->smallChart->graph(0)->setPen(pen);

    ui->smallChart->replot();

    number max = 0;
    for (size_t i = 0; i < fRunEpochsL.size(); i++) {
        if (fRunEpochsL[i] > max) {
            max = fRunEpochsL[i];
        }
        if (fValidateEpochsL[i] > max) {
            max = fValidateEpochsL[i];
        }
    }
    max = max * 1.1;

    ui->smallChart->yAxis->setRange(0, max);
    ui->smallChart->xAxis->setRange(1, iEp + 1);
    ui->smallChart->replot();
}

void ValidationWidget::renew()
{
    size_t currentEpoch = supervisor->curEpoch;
    //qDebug() << currentEpoch;
    size_t curTraining = supervisor->curTraining;
    size_t curValidation = supervisor->curValidation;

    this->ui->pro_Epoch->setValue(currentEpoch);
    this->ui->pro_Training->setValue(curTraining + curValidation);
    if (currentEpoch == supervisor->szEpoch) {
        this->drawLoss(supervisor->szEpoch -1);
        this->drawHistogram(supervisor->szEpoch -1);
        process->join();
        delete process;
        process = nullptr;
        timer->stop();
        QMessageBox::information(this, tr("end"), tr("End of training"), QMessageBox::Ok);

        this->ui->sb_szBatches->setEnabled(true);
        this->ui->sb_szTraining->setEnabled(true);
        this->ui->sb_szValidation->setEnabled(true);
        this->ui->btn_loadData->setEnabled(true);

    } else{
        this->drawLoss(supervisor->curEpoch);
        this->drawHistogram(this->supervisor->curEpoch);
    }
}
