#ifndef CONVERTWIDGET_H
#define CONVERTWIDGET_H

#include <QWidget>
#include <module.h>
#include <qgprawdata.h>
#include <thread>

namespace Ui {
class ConvertWidget;
}

class ConvertWidget : public QWidget
{
    Q_OBJECT
public:
    std::thread * worker = nullptr;
    QTimer *timer;
    Module * module;
    QGPRawData converter;

public:
    explicit ConvertWidget(QWidget *parent = nullptr);
    ~ConvertWidget();

    void setEventsPerFile(int number);

    void setNumberOfFiles(int number);

    void setDrawType(int number);

    void drawMomentum();

    void drawPolar();

    void drawInclination();


    void renew();

    void convert();

    void loadInputPath();

    void loadOutputPath();

private:
    Ui::ConvertWidget *ui;
};

#endif // CONVERTWIDGET_H
