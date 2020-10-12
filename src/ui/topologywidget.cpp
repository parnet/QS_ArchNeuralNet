#include "topologywidget.h"
#include "ui_topologywidget.h"

TopologyWidget::TopologyWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TopologyWidget)
{
    ui->setupUi(this);
}

TopologyWidget::~TopologyWidget()
{
    delete ui;
}
