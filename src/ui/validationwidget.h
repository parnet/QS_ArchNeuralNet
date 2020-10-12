#ifndef VALIDATIONWIDGET_H
#define VALIDATIONWIDGET_H

#include <QWidget>
#include <supervisor.h>
#include <thread>

namespace Ui {
class ValidationWidget;
}

class ValidationWidget : public QWidget
{
    Q_OBJECT
private:
    Ui::ValidationWidget *ui;

public:
    QTimer * timer;
    Supervisor * supervisor;
    QString pathDataset;
    size_t szEpoch;
    std::thread *process = nullptr;

public:
    explicit ValidationWidget(QWidget *parent = nullptr);
    ~ValidationWidget();

public slots:

    void loadNeuralNet();

    void saveNeuralNet(); // todo delete?

    void loadDataset();

    void analyzeData();

    void setOffset(int number);

    void setTestdata(int number);

    void setBatchsize(int number);

    void drawHistogram(size_t cEpoch = 0);

    void drawLoss(size_t cEpoch = 0);

    void renew();


};

#endif // VALIDATIONWIDGET_H
