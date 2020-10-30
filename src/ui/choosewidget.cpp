#include "choosewidget.h"
#include "ui_choosewidget.h"
#include "mainwindow.h"

ChooseWidget::ChooseWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ChooseWidget)
{
    ui->setupUi(this);
    MainWindow * main = static_cast<MainWindow*>(this->parentWidget());

    if (main != nullptr) {
    connect(this->ui->btn_newTraining,SIGNAL(clicked()), main, SLOT(changeTabTraining()));
    connect(this->ui->btn_newConversion,SIGNAL(clicked()), main, SLOT(changeTabConvert()));
    connect(this->ui->btn_newValidation,SIGNAL(clicked()), main, SLOT(changeTabValidation()));
    connect(this->ui->btn_newPlots, SIGNAL(clicked()), main, SLOT(changeTabPlotting()));
    }

}

ChooseWidget::~ChooseWidget()
{
    delete ui;
}

void ChooseWidget::setParent(QWidget *parent)
{
    MainWindow * main = static_cast<MainWindow*>(parent);

    if (parent != nullptr) {
        connect(this->ui->btn_newTraining,SIGNAL(clicked()), main, SLOT(changeTabTraining()));
        connect(this->ui->btn_newConversion,SIGNAL(clicked()), main, SLOT(changeTabConvert()));
        connect(this->ui->btn_newValidation,SIGNAL(clicked()), main, SLOT(changeTabValidation()));
        connect(this->ui->btn_newPlots,SIGNAL(clicked()), main, SLOT(changeTabPlotting()));
    }
    QWidget::setParent(parent);
}
