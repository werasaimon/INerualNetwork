QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets printsupport

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    NerualNetwork/IGenetic.cpp \
    NerualNetwork/ILayerNeurons.cpp \
    NerualNetwork/INerualNetwork.cpp \
    main.cpp \
    paint.cpp \
    qcustomplot.cpp

HEADERS += \
    NerualNetwork/IGenetic.h \
    NerualNetwork/ILayerNeurons.h \
    NerualNetwork/INerualNetwork.h \
    paint.h \
    qcustomplot.h

FORMS += \
    paint.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
