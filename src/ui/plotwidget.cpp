#include "plotwidget.h"
#include "ui_plotwidget.h"

PlotWidget::PlotWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PlotWidget),
    supervisor(new Supervisor(new Module()))
{
    ui->setupUi(this);
    connect(ui->btn_addPlotData, SIGNAL(clicked()), this, SLOT(addPlotData()));
    connect(ui->btn_resetPlotArea, SIGNAL(clicked()), this, SLOT(resetPlotData()));
    connect(ui->btn_exportToPNG, SIGNAL(clicked()), this, SLOT(exportToPNG()));
    connect(ui->sl_TrainingR, SIGNAL(valueChanged(int)), this, SLOT(changeColorTrainingR(int)));
    connect(ui->sl_TrainingG, SIGNAL(valueChanged(int)), this, SLOT(changeColorTrainingG(int)));
    connect(ui->sl_TrainingB, SIGNAL(valueChanged(int)), this, SLOT(changeColorTrainingB(int)));
    connect(ui->sl_ValidationR, SIGNAL(valueChanged(int)), this, SLOT(changeColorValidationR(int)));
    connect(ui->sl_ValidationG, SIGNAL(valueChanged(int)), this, SLOT(changeColorValidationG(int)));
    connect(ui->sl_ValidationB, SIGNAL(valueChanged(int)), this, SLOT(changeColorValidationB(int)));
    connect(ui->tGraphNameField, SIGNAL(textChanged(QString)), this, SLOT(setTGraphName(QString)));
    connect(ui->vGraphNameField, SIGNAL(textChanged(QString)), this, SLOT(setVGraphName(QString)));
    connect(ui->sb_yMin, SIGNAL(valueChanged(int)), this, SLOT(changeYmin(int)));
    connect(ui->sb_yMax, SIGNAL(valueChanged(int)), this, SLOT(changeYmax(int)));
    connect(ui->cb_tGraphDesign, SIGNAL(currentIndexChanged(int)), this, SLOT(changeTGraphDesign(int)));
    connect(ui->cb_vGraphDesign, SIGNAL(currentIndexChanged(int)), this, SLOT(changeVGraphDesign(int)));

    ui->cb_tGraphDesign->addItem("None");
    ui->cb_tGraphDesign->addItem("Dot");
    ui->cb_tGraphDesign->addItem("Cross");
    ui->cb_tGraphDesign->addItem("Plus");
    ui->cb_tGraphDesign->addItem("Circle");
    ui->cb_tGraphDesign->addItem("Disc");
    ui->cb_tGraphDesign->addItem("Square");
    ui->cb_tGraphDesign->addItem("Diamond");
    ui->cb_tGraphDesign->addItem("Star");
    ui->cb_tGraphDesign->addItem("Triangle");
    ui->cb_tGraphDesign->addItem("Invert. Triangle");

    ui->cb_vGraphDesign->addItem("None");
    ui->cb_vGraphDesign->addItem("Dot");
    ui->cb_vGraphDesign->addItem("Cross");
    ui->cb_vGraphDesign->addItem("Plus");
    ui->cb_vGraphDesign->addItem("Circle");
    ui->cb_vGraphDesign->addItem("Disc");
    ui->cb_vGraphDesign->addItem("Square");
    ui->cb_vGraphDesign->addItem("Diamond");
    ui->cb_vGraphDesign->addItem("Star");
    ui->cb_vGraphDesign->addItem("Triangle");
    ui->cb_vGraphDesign->addItem("Invert. Triangle");

    QCPLayoutGrid *subLayout = new QCPLayoutGrid;
    QCPLayoutElement *dummyElement = new QCPLayoutElement;
    ui->plotArea->plotLayout()->addElement(0, 1, subLayout); // add sub-layout in the cell to the right of the main axis rect
    ui->plotArea->plotLayout()->setColumnStretchFactor(1, 0.1);
    subLayout->addElement(0, 0, ui->plotArea->legend); // add legend
    subLayout->addElement(1, 0, dummyElement); // add dummy element below legend
    this->tGraphStyle.setSize(10.0);
    this->vGraphStyle.setSize(10.0);

    resetPlotData();
}

PlotWidget::~PlotWidget()
{
    delete ui;
}


void PlotWidget::addPlotData()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Add Plot Data"), "", tr("NeuralNetPlot (*.nnp);;All Files (*)"));

    if(fileName == ""){
        return;
    }
    std::string str = fileName.toStdString();
    std::ifstream file(str.c_str());
    double value;
    int nEpochs;
    file >> nEpochs;
    this->xMaxRange = (nEpochs > this->xMaxRange) ? nEpochs : this->xMaxRange;
    std::vector<double> tmpVec;
    std::vector<std::vector<double>> tmpCollectVec;
    for(size_t line = 0; line < 4; line++)
    {
        tmpVec.clear();
        for(int val = 0; val < nEpochs; val++)
        {
            file >> value;
            tmpVec.push_back(value);
        }
        tmpCollectVec.push_back(tmpVec);
    }

    std::vector<double> totalTrainingValues = tmpCollectVec[0];
    std::vector<double> correctTrainingValues= tmpCollectVec[1];
    std::vector<double> totalValidationValues= tmpCollectVec[2];
    std::vector<double> correctValidationValues= tmpCollectVec[3];
    QVector<double> trainingAccuracies;
    QVector<double> trainingEpochs;
    QVector<double> validationAccuracies;
    QVector<double> validationEpochs;
    tmpCollectVec.clear();

    double acc = 0.0;
    for(size_t i = 0; i < totalTrainingValues.size(); i++)
    {
        acc = correctTrainingValues[i] / totalTrainingValues[i];
        trainingAccuracies.push_back(acc * 100);
        acc = correctValidationValues[i] / totalValidationValues[i];
        validationAccuracies.push_back(acc * 100);
        trainingEpochs.push_back(double(i));
        validationEpochs.push_back(double(i));
    }

    QFont legendFont = font();
    legendFont.setPointSize(8);
    ui->plotArea->legend->setFont(legendFont);
    ui->plotArea->legend->setVisible(true);

    QPen pen;
    pen.setWidth(2);
    ui->plotArea->xAxis->setRange(0, this->xMaxRange);
    ui->plotArea->yAxis->setRange(this->yMinRange, this->yMaxRange);

    QCPGraph *tGraph = ui->plotArea->addGraph();
    tGraph->setData(trainingEpochs, trainingAccuracies);
    tGraph->setScatterStyle(this->tGraphStyle);
    tGraph->setName(this->tGraphName);
    tGraph->addToLegend();
    pen.setColor(QColor(this->color1[0], this->color1[1], this->color1[2]));
    pen.setStyle(Qt::DashLine);
    tGraph->setPen(pen);

    QCPGraph *vGraph = ui->plotArea->addGraph();
    vGraph->setData(validationEpochs, validationAccuracies);
    vGraph->setScatterStyle(this->vGraphStyle);
    vGraph->setName(this->vGraphName);
    vGraph->addToLegend();
    pen.setColor(QColor(this->color2[0], this->color2[1], this->color2[2]));
    pen.setStyle(Qt::DashDotDotLine);
    vGraph->setPen(pen);

    this->nGraphs += 2;

    ui->plotArea->replot();

}

void PlotWidget::resetPlotData()
{
    // reset colors to default
    QPalette palette1 = ui->label_ColorTraining->palette();
    palette1.setColor(QPalette::WindowText, this->defaultTGraphColor);
    ui->label_ColorTraining->setPalette(palette1);
    QPalette palette2 = ui->label_ColorTraining->palette();
    palette2.setColor(QPalette::WindowText, this->defaultVGraphColor);
    ui->label_ColorValidation->setPalette(palette2);

    // clear graph name fields
    ui->tGraphNameField->insert("");
    ui->vGraphNameField->insert("");

    // clear plotted graphs
    ui->plotArea->clearGraphs();
    ui->plotArea->xAxis->setLabel("Number of Epochs");
    ui->plotArea->yAxis->setLabel("Accuracy (%)");
    ui->plotArea->xAxis->setRange(0, 1);
    ui->plotArea->yAxis->setRange(this->yMinRange, this->yMaxRange);
    ui->plotArea->legend->setVisible(false);
    ui->plotArea->replot();
    this->nGraphs = 0;
}


void PlotWidget::changeColorTrainingR(int value)
{
    this->color1[0] = value;
    QColor tGraphColor = QColor(this->color1[0], this->color1[1], this->color1[2]);
    QPalette palette = ui->label_ColorTraining->palette();
    palette.setColor(QPalette::WindowText, tGraphColor);
    ui->label_ColorTraining->setPalette(palette);
}

void PlotWidget::changeColorTrainingG(int value)
{
    this->color1[1] = value;
    QColor tGraphColor = QColor(this->color1[0], this->color1[1], this->color1[2]);
    QPalette palette = ui->label_ColorTraining->palette();
    palette.setColor(QPalette::WindowText, tGraphColor);
    ui->label_ColorTraining->setPalette(palette);
}

void PlotWidget::changeColorTrainingB(int value)
{
    this->color1[2] = value;
    QColor tGraphColor = QColor(this->color1[0], this->color1[1], this->color1[2]);
    QPalette palette = ui->label_ColorTraining->palette();
    palette.setColor(QPalette::WindowText, tGraphColor);
    ui->label_ColorTraining->setPalette(palette);
}

void PlotWidget::changeColorValidationR(int value)
{
    this->color2[0] = value;
    QColor vGraphColor = QColor(this->color2[0], this->color2[1], this->color2[2]);
    QPalette palette = ui->label_ColorValidation->palette();
    palette.setColor(QPalette::WindowText, vGraphColor);
    ui->label_ColorValidation->setPalette(palette);
}

void PlotWidget::changeColorValidationG(int value)
{
    this->color2[1] = value;
    QColor vGraphColor = QColor(this->color2[0], this->color2[1], this->color2[2]);
    QPalette palette = ui->label_ColorValidation->palette();
    palette.setColor(QPalette::WindowText, vGraphColor);
    ui->label_ColorValidation->setPalette(palette);
}

void PlotWidget::changeColorValidationB(int value)
{
    this->color2[2] = value;
    QColor vGraphColor = QColor(this->color2[0], this->color2[1], this->color2[2]);
    QPalette palette = ui->label_ColorValidation->palette();
    palette.setColor(QPalette::WindowText, vGraphColor);
    ui->label_ColorValidation->setPalette(palette);
}

void PlotWidget::setTGraphName(QString str)
{
    this->tGraphName = str;
}

void PlotWidget::setVGraphName(QString str)
{
    this->vGraphName = str;
}

void PlotWidget::changeYmax(int value)
{
    this->yMaxRange = value;
    ui->plotArea->yAxis->setRange(this->yMinRange, this->yMaxRange);
    ui->plotArea->replot();
}

void PlotWidget::changeYmin(int value)
{
    this->yMinRange = value;
    ui->plotArea->yAxis->setRange(this->yMinRange, this->yMaxRange);
    ui->plotArea->replot();
}

void PlotWidget::changeTGraphDesign(int value)
{
    switch(value)
    {
        case 0:
                this->tGraphStyle = QCPScatterStyle::ssNone;
                break;
        case 1:
                this->tGraphStyle = QCPScatterStyle::ssDot;
                break;
        case 2:
                this->tGraphStyle = QCPScatterStyle::ssCross;
                break;
        case 3:
                this->tGraphStyle = QCPScatterStyle::ssPlus;
                break;
        case 4:
                this->tGraphStyle = QCPScatterStyle::ssCircle;
                break;
        case 5:
                this->tGraphStyle = QCPScatterStyle::ssDisc;
                break;
        case 6:
                this->tGraphStyle = QCPScatterStyle::ssSquare;
                break;
        case 7:
                this->tGraphStyle = QCPScatterStyle::ssDiamond;
                break;
        case 8:
                this->tGraphStyle = QCPScatterStyle::ssStar;
                break;
        case 9:
                this->tGraphStyle = QCPScatterStyle::ssTriangle;
                break;
        case 10:
                this->tGraphStyle = QCPScatterStyle::ssTriangleInverted;
                break;
        default:
            this->vGraphStyle = QCPScatterStyle::ssNone;
    }
}

void PlotWidget::changeVGraphDesign(int value)
{
    switch(value)
    {
        case 0:
                this->vGraphStyle = QCPScatterStyle::ssNone;
                break;
        case 1:
                this->vGraphStyle = QCPScatterStyle::ssDot;
                break;
        case 2:
                this->vGraphStyle = QCPScatterStyle::ssCross;
                break;
        case 3:
                this->vGraphStyle = QCPScatterStyle::ssPlus;
                break;
        case 4:
                this->vGraphStyle = QCPScatterStyle::ssCircle;
                break;
        case 5:
                this->vGraphStyle = QCPScatterStyle::ssDisc;
                break;
        case 6:
                this->vGraphStyle = QCPScatterStyle::ssSquare;
                break;
        case 7:
                this->vGraphStyle = QCPScatterStyle::ssDiamond;
                break;
        case 8:
                this->vGraphStyle = QCPScatterStyle::ssStar;
                break;
        case 9:
                this->vGraphStyle = QCPScatterStyle::ssTriangle;
                break;
        case 10:
                this->vGraphStyle = QCPScatterStyle::ssTriangleInverted;
                break;
        default:
                this->vGraphStyle = QCPScatterStyle::ssNone;
    }
    std::cout << "changed" << std::endl;
}


void PlotWidget::exportToPNG()
{
    QFileDialog *dialog = new QFileDialog();
    dialog->setDefaultSuffix("png");
    QString fileName = dialog->getSaveFileName(this, tr("Export to .png"), "", tr("PNG (*.png);;All Files (*)"));
    std::ofstream file(fileName.toStdString().c_str());
    delete dialog;
    ui->plotArea->QCustomPlot::savePng(fileName, 1600, 900, 1.0, -1);
}
